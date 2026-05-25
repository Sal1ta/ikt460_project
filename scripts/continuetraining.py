# Keep training an existing Afterstate model

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.afterstate import AfterstateSearchAgent, AfterstateValueAgent
from src.agents import GreedyAgent, HomeFirstRandomAgent, MinimaxAgent
from src.env import ChineseCheckersEnv
from src.paths import (
    AFTERSTATE_BACKUP_DIR,
    AFTERSTATE_TWOPLAYER_MODEL,
    AFTERSTATE_CHECKPOINT_DIR,
    AFTERSTATE_EXTERNAL_MODEL,
    AFTERSTATE_TRAINED_MODEL,
    ensure_project_dirs,
)
from src.perspective import PLAYABLE_COLOR_PAIRS, color_pair_for_game
from src.rewards import (
    afterstate_shaped_reward,
    compute_potential,
    evaluate_afterstate,
    evaluate_afterstate_by_pair,
    model_selection_score,
    weakest_afterstate_pair_summary,
)

ensure_project_dirs()

# Defaults are smaller because the model already knows how to play
CONTINUE_EPISODES = 300
DEMO_EPISODES = 60
MAX_TURNS = 250
EVAL_FREQ = 20
EVAL_GAMES = 6
PAIR_EVAL_GAMES = 2
CHECKPOINT_FREQ = 250
TARGET_UPDATE_STEPS = 500
WEAKEST_PAIR_MARGIN = 20.0
# Depth 1 keeps training fast
MINIMAX_DEPTH = 1

# Demo games use more teacher help Later games use harder opponents
DEMO_OPPONENT_MIX = (
    ("easy",    0.35),
    ("greedy",  0.45),
    ("minimax", 0.20),
)
EARLY_TRAINING_OPPONENT_MIX = (
    ("easy",       0.20),
    ("greedy",     0.45),
    ("minimax",    0.30),
    ("historical", 0.05),
)
LATE_TRAINING_OPPONENT_MIX = (
    ("easy",       0.05),
    ("greedy",     0.30),
    ("minimax",    0.30),
    ("historical", 0.35),
)
HISTORICAL_POOL_LIMIT = 4
CURRICULUM_EPISODES = 120

# New model files are saved as backups first Final files change only after tests pass
CHALLENGER_BEST_MODEL = str(AFTERSTATE_BACKUP_DIR / "challengermixedminimaxselected.pth")
CHALLENGER_EXTERNAL_MODEL = str(AFTERSTATE_BACKUP_DIR / "challengermixedminimaxexternal.pth")
CHALLENGER_TRAINED_MODEL = str(AFTERSTATE_BACKUP_DIR / "challengermixedminimaxtrained.pth")

shaped_reward = afterstate_shaped_reward
evaluate = evaluate_afterstate

def parse_args():
    # Only show the settings we actually change
    parser = argparse.ArgumentParser(description="Continue training the Afterstate value agent safely.")
    parser.add_argument("--episodes", type=int, default=CONTINUE_EPISODES, help="Continuation training episodes.")
    parser.add_argument("--demo-games", type=int, default=DEMO_EPISODES, help="Fresh teacher-demo games before training.")
    parser.add_argument("--minimax-depth", type=int, default=MINIMAX_DEPTH, help="Minimax opponent depth during continuation.")
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ, help="Evaluate every N episodes.")
    parser.add_argument("--eval-games", type=int, default=EVAL_GAMES, help="Games per quick eval checkpoint.")
    parser.add_argument("--pair-eval-games", type=int, default=PAIR_EVAL_GAMES, help="Games per lane during pair eval.")
    parser.add_argument("--no-promote", action="store_true", help="Never copy the challenger over tournament checkpoints.")
    return parser.parse_args()

def choose_training_opponent(opponents, mix):
    # Pick opponents from the chosen training mix
    labels = []
    weights = []
    for name, weight in mix:
        candidate = opponents.get(name)
        if name == "historical":
            # Old models are optional Skip this if none are loaded
            if not candidate:
                continue
        elif candidate is None:
            continue
        labels.append(name)
        weights.append(weight)

    label = random.choices(labels, weights=weights, k=1)[0]
    if label == "historical":
        # Each old model has a name and a fixed agent
        candidate = random.choice(opponents[label])
        return candidate["name"], candidate["agent"]
    return label, opponents[label]

def training_mix_for_episode(episode):
    # Start easier, then use more old models later
    if int(episode) < CURRICULUM_EPISODES:
        return EARLY_TRAINING_OPPONENT_MIX
    return LATE_TRAINING_OPPONENT_MIX

def load_historical_agent(state_size, model_path, name):
    # Use fixed old models, not the model currently being trained
    candidate = AfterstateValueAgent(state_size=state_size, player_color="yellow", name=name)
    if not candidate.load_model(str(model_path), verbose=False):
        return None
    candidate.epsilon = 0.0
    return candidate

def historical_candidate_paths(exclude_paths=None, limit=HISTORICAL_POOL_LIMIT):
    # Old models are useful because they play a bit differently
    exclude = {
        str(Path(path).resolve())
        for path in (exclude_paths or [])
        if path is not None
    }
    seen = set()
    candidates = []

    ordered_paths = [
        # Try the main model files first, then backup files
        AFTERSTATE_EXTERNAL_MODEL,
        AFTERSTATE_TWOPLAYER_MODEL,
        AFTERSTATE_TRAINED_MODEL,
        *sorted(AFTERSTATE_CHECKPOINT_DIR.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True),
        *sorted(AFTERSTATE_BACKUP_DIR.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True),
    ]

    for path in ordered_paths:
        candidate = Path(path)
        if not candidate.exists():
            continue
        resolved = str(candidate.resolve())
        if resolved in exclude or resolved in seen:
            # Do not train against the same model file we are changing
            continue
        seen.add(resolved)
        candidates.append(candidate)
        if len(candidates) >= int(limit):
            break

    return candidates

def build_historical_opponent_pool(state_size, exclude_paths=None, limit=HISTORICAL_POOL_LIMIT):
    # Keep the old model pool small so training stays fast
    pool = []
    for index, path in enumerate(historical_candidate_paths(exclude_paths=exclude_paths, limit=limit)):
        candidate = load_historical_agent(
            state_size,
            path,
            name=f"Hist:{path.stem}",
        )
        if candidate is None:
            # Skip model files that cannot load
            continue
        pool.append({
            "name": candidate.name,
            "path": str(path),
            "agent": candidate,
            "index": index,
        })
    return pool

def copy_if_different(src, dst):
    # Do not copy a file onto itself
    source = Path(src)
    target = Path(dst)
    if not source.exists():
        return False
    if source.resolve() == target.resolve():
        return False
    # Keep file times when copying
    shutil.copy2(source, target)
    return True

def load_search_teacher(state_size, model_path):
    # Use the strongest saved search model as the teacher when possible
    if not model_path or not os.path.exists(model_path):
        return None
    teacher = AfterstateSearchAgent(
        state_size=state_size,
        player_color="yellow",
        name="AfterstateSearchTeacher",
    )
    if not teacher.load_model(str(model_path), verbose=False):
        return None
    teacher.epsilon = 0.0
    # Teacher should not use random moves
    return teacher

def choose_teacher_action(env, valid_actions, player_color="yellow", teacher=None):
    if not valid_actions:
        return None

    if teacher is not None:
        # Use the saved search teacher if it returns a legal move
        action = teacher.choose_action(env, valid_actions)
        if action is not None:
            return action

    # Teacher choices should be the same each run
    best_action = None
    best_score = None
    best_index = None

    for action in valid_actions:
        score = env.evaluate_action_progress(action, player_color)
        action_index = int(action[0]) * 121 + int(action[1])
        # If moves look equal, choose the same one each time
        if best_score is None or score > best_score or (score == best_score and action_index < best_index):
            best_action = action
            best_score = score
            best_index = action_index

    return best_action

def collect_demo_experiences(agent, opponents, teacher=None, episodes=120):
    # Add teacher examples before harder training
    print(f"\nCollecting mixed demo experiences   {episodes} games")
    wins = 0

    for episode in range(episodes):
        # Demo games use their own opponent mix
        _, opponent = choose_training_opponent(opponents, DEMO_OPPONENT_MIX)
        player_colors = color_pair_for_game(episode)
        agent_color = player_colors[0]
        env = ChineseCheckersEnv(num_players=2, player_colors=player_colors, max_turns=MAX_TURNS)
        env.reset()
        done = False
        info = {}
        steps = 0

        while not done and steps < MAX_TURNS:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            if env.get_current_player() != agent_color:
                # Let the opponent answer before saving the example
                action = opponent.choose_action(env, valid_actions)
                if action is None:
                    break
                _, _, done, info = env.step(action)
                steps += 1
                continue

            action = choose_teacher_action(env, valid_actions, agent_color, teacher=teacher)
            if action is None:
                break

            phi_s = compute_potential(env, agent_color)
            progress = env.evaluate_action_progress(action, agent_color)
            afterstate = agent.afterstate_from_env_action(env, action, agent_color)
            _, _, done, info = env.step(action)
            steps += 1

            if not done:
                # Save after the opponent reply, so it is our turn again
                opponent_actions = env.get_valid_actions()
                if opponent_actions:
                    opponent_action = opponent.choose_action(env, opponent_actions)
                    if opponent_action is not None:
                        _, _, done, info = env.step(opponent_action)
                        steps += 1

            phi_sp = compute_potential(env, agent_color)
            reward = shaped_reward(done, info, agent_color, phi_s, phi_sp, progress)
            next_position_state = agent.position_state_from_env(env, agent_color) if not done else None
            # Put teacher moves in demo memory
            agent.remember(afterstate, reward, next_position_state, done, demo=True)

        if f"{agent_color} wins" in info.get("message", ""):
            wins += 1

        agent.finish_episode()

        if (episode + 1) % 30 == 0:
            print(f"  {episode + 1}/{episodes}   demo {len(agent.demo_memory)}   wins {wins}/{episode + 1}")

    print(f"  Done   {len(agent.demo_memory)} demo experiences   wins {wins}/{episodes}")

def main():
    global CONTINUE_EPISODES, DEMO_EPISODES, EVAL_FREQ, EVAL_GAMES, PAIR_EVAL_GAMES, MINIMAX_DEPTH

    # Keep command line numbers valid
    args = parse_args()
    CONTINUE_EPISODES = max(1, int(args.episodes))
    DEMO_EPISODES = max(0, int(args.demo_games))
    EVAL_FREQ = max(1, int(args.eval_freq))
    EVAL_GAMES = max(1, int(args.eval_games))
    PAIR_EVAL_GAMES = max(1, int(args.pair_eval_games))
    MINIMAX_DEPTH = max(1, int(args.minimax_depth))
    promotion_enabled = not args.no_promote

    print(f"\nAfterstate Continuation   (vs Greedy + Minimax depth {MINIMAX_DEPTH})\n")

    # Ask the environment how large the state is
    env_probe = ChineseCheckersEnv(num_players=2, player_colors=["yellow", "purple"])
    state_size = len(env_probe.reset())

    agent = AfterstateValueAgent(state_size=state_size, player_color="yellow", name="AfterstateAgent")
    agent.demo_fraction = 0.70

    loaded = False
    loaded_checkpoint_path = None
    # Try loading the newest model first, then older public model files
    for candidate in (
        str(AFTERSTATE_TRAINED_MODEL),
        str(AFTERSTATE_TWOPLAYER_MODEL),
        str(AFTERSTATE_EXTERNAL_MODEL),
    ):
        if os.path.exists(candidate) and agent.load_model(candidate):
            loaded = True
            loaded_checkpoint_path = candidate
            break
    if not loaded:
        print(f"ERROR: could not load an afterstate checkpoint from {AFTERSTATE_TRAINED_MODEL} or {AFTERSTATE_TWOPLAYER_MODEL}")
        return

    # Use small learning steps so we do not ruin a good model
    agent.epsilon = 0.02
    agent.epsilon_decay = 0.99997
    agent.epsilon_min = 0.005
    agent.learning_rate = 0.000001
    # Rebuild the optimizer after changing the learning rate
    agent.optimizer = agent.optimizer.__class__(agent.value_network.parameters(), lr=agent.learning_rate)
    agent.train_updates = 0

    print(f"  State size    {state_size}")
    print(f"  Epsilon       {agent.epsilon:.2f} -> {agent.epsilon_min:.2f}   decay {agent.epsilon_decay}")
    print(f"  Learning rate {agent.learning_rate}")
    print(f"  Opponent mix  early:  20% Easy   45% Greedy   30% Minimax    5% historical")
    print(f"                late:    5% Easy   30% Greedy   30% Minimax   35% historical")
    print(f"  Color lanes   {', '.join(f'{a}/{b}' for a, b in PLAYABLE_COLOR_PAIRS)}")
    print(f"  Memory        online {len(agent.memory)} (checkpoint stores weights only)   demo {len(agent.demo_memory)}")
    print(f"  Targets       {agent.n_step}-step bootstrap")
    print(f"  Pair eval     Greedy by lane   {PAIR_EVAL_GAMES} games per pair")
    print(f"  Lane floor    best weakest lane - {WEAKEST_PAIR_MARGIN:.1f} score")
    if not promotion_enabled:
        print("  Promotion     disabled (--no-promote)")

    greedy_opp = GreedyAgent(name="GreedyOpponent")
    minimax_opp = MinimaxAgent(name="MinimaxOpponent", depth=MINIMAX_DEPTH)
    easy_opp = HomeFirstRandomAgent(name="EasyOpponent")

    # Test before training so we know the starting score
    baseline_greedy_wins, baseline_greedy_pins = evaluate(agent, greedy_opp, games=EVAL_GAMES, max_turns=MAX_TURNS)
    baseline_easy_wins, baseline_easy_pins = evaluate(agent, easy_opp, games=EVAL_GAMES, max_turns=500)
    baseline_pair_eval = evaluate_afterstate_by_pair(
        agent,
        greedy_opp,
        games_per_pair=PAIR_EVAL_GAMES,
        max_turns=MAX_TURNS,
    )
    baseline_weakest = weakest_afterstate_pair_summary(baseline_pair_eval["pairs"])
    best_eval_score = (
        model_selection_score(baseline_greedy_wins, baseline_greedy_pins)
        + 0.15 * model_selection_score(baseline_easy_wins, baseline_easy_pins)
    )
    best_eval_episode = 0
    best_weakest_score = baseline_weakest["score"]
    best_weakest_pair = baseline_weakest["pair"]
    best_external_score = best_eval_score
    current_best_candidate_path = loaded_checkpoint_path
    current_best_candidate_is_challenger = False

    # The loaded model is the score to beat
    print(f"  Baseline     Greedy {baseline_greedy_wins:.1f}%   pins {baseline_greedy_pins:.1f}   score {best_eval_score:.1f}")
    print(f"  Weakest lane {baseline_weakest['pair']}   score {baseline_weakest['score']:.1f}")

    historical_pool = build_historical_opponent_pool(
        state_size,
        exclude_paths=(
            # Do not add models from this same run as opponents
            loaded_checkpoint_path,
            CHALLENGER_BEST_MODEL,
            CHALLENGER_EXTERNAL_MODEL,
            CHALLENGER_TRAINED_MODEL,
        ),
        limit=HISTORICAL_POOL_LIMIT,
    )
    if historical_pool:
        print(f"  Historical pool   {len(historical_pool)} checkpoints")
    else:
        print("  Historical pool   none yet")

    teacher_agent = load_search_teacher(state_size, loaded_checkpoint_path)
    if teacher_agent is not None:
        print("  Teacher       frozen Afterstate+Search from loaded checkpoint")
    else:
        print("  Teacher       fallback progress teacher")

    training_opponents = {
        # These are the opponent pools used during training
        "greedy":    GreedyAgent(name="GreedyOpponent"),
        "minimax":   MinimaxAgent(name="MinimaxOpponent", depth=MINIMAX_DEPTH),
        "easy":      HomeFirstRandomAgent(name="EasyTrainingOpponent"),
        "historical": historical_pool,
    }

    collect_demo_experiences(agent, training_opponents, teacher=teacher_agent, episodes=DEMO_EPISODES)

    # Use less teacher help over time
    teacher_guidance_episodes = 200
    teacher_start_prob = 0.35
    # Stop early if the model gets clearly worse
    early_stop_patience = 3
    min_episodes_before_early_stop = EVAL_FREQ
    collapse_stop_margin = 30.0
    hard_collapse_win_margin = 20.0
    hard_collapse_pin_margin = 1.5

    evals_without_improvement = 0
    baseline_eval_score = best_eval_score

    print(f"\nTraining   {CONTINUE_EPISODES} episodes vs Greedy + Minimax(depth={MINIMAX_DEPTH})\n")
    start_time = time.time()
    recent_wins = []
    recent_rewards = []
    recent_pins = []
    print_freq = 20

    for episode in range(CONTINUE_EPISODES):
        # Pick easier or harder opponents based on episode number
        opponent_label, opponent = choose_training_opponent(
            training_opponents,
            training_mix_for_episode(episode),
        )
        player_colors = color_pair_for_game(episode)
        agent_color = player_colors[0]
        # Keep changing colour pairs so both directions get trained
        env = ChineseCheckersEnv(num_players=2, player_colors=player_colors, max_turns=MAX_TURNS)

        env.reset()
        done = False
        info = {}
        total_reward = 0.0
        steps = 0
        agent_step = 0
        won = False
        # Remember each pins last cell to punish simple backtracking
        # This matches the move pattern used by the tournament player
        pin_last_origin: dict = {}

        while not done and steps < MAX_TURNS:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            if env.get_current_player() == agent_color:
                # Still learn from the move when teacher help picked it
                teacher_prob = 0.0
                if episode < teacher_guidance_episodes:
                    teacher_prob = teacher_start_prob * (1.0 - episode / teacher_guidance_episodes)

                guided = False
                if teacher_prob > 0.0 and random.random() < teacher_prob:
                    # Teacher picked moves go into demo memory
                    action = choose_teacher_action(env, valid_actions, agent_color, teacher=teacher_agent)
                    guided = action is not None
                else:
                    action = agent.choose_action(env, valid_actions)

                if action is None:
                    break

                pin_id, destination = action
                # Save score values before changing the board
                phi_s = compute_potential(env, agent_color)
                progress = env.evaluate_action_progress(action, agent_color)
                afterstate = agent.afterstate_from_env_action(env, action, agent_color)

                # Remember where this pin started
                current_position = None
                for pin in env.pins_on_board:
                    if str(pin.color) == str(agent_color) and int(pin.id) == int(pin_id):
                        current_position = int(pin.axialindex)
                        break

                _, _, done, info = env.step(action)
                steps += 1

                if not done:
                    # Let one opponent move before saving the next state
                    opponent_actions = env.get_valid_actions()
                    if opponent_actions:
                        opponent_action = opponent.choose_action(env, opponent_actions)
                        if opponent_action is not None:
                            _, _, done, info = env.step(opponent_action)
                            steps += 1

                phi_sp = compute_potential(env, agent_color)
                reward = shaped_reward(done, info, agent_color, phi_s, phi_sp, progress)

                # Small penalty for moving a pin straight back
                if current_position is not None:
                    last_origin = pin_last_origin.get(int(pin_id))
                    if last_origin is not None and int(destination) == last_origin:
                        # This move is still legal, just less attractive
                        reward -= 0.20
                    pin_last_origin[int(pin_id)] = current_position

                next_position_state = agent.position_state_from_env(env, agent_color) if not done else None
                agent.remember(afterstate, reward, next_position_state, done, demo=guided)
                total_reward += reward

                agent_step += 1
                if agent_step % 5 == 0:
                    # Replay every few moves so training stays fast
                    did_replay = agent.replay()
                    if did_replay and agent.train_updates % TARGET_UPDATE_STEPS == 0:
                        agent.update_target_network()

                if done and f"{agent_color} wins" in info.get("message", ""):
                    won = True
            else:
                action = opponent.choose_action(env, valid_actions)
                if action is None:
                    break
                _, _, done, info = env.step(action)
                steps += 1

        if agent.epsilon > agent.epsilon_min:
            # Reduce exploration slowly
            agent.epsilon *= agent.epsilon_decay

        agent.finish_episode()

        recent_wins.append(1 if won else 0)
        recent_rewards.append(total_reward)
        recent_pins.append(env.count_player_pins_in_target(agent_color))

        if (episode + 1) % CHECKPOINT_FREQ == 0:
            agent.save_model(str(AFTERSTATE_CHECKPOINT_DIR / f"continueep{episode + 1}.pth"))

        if (episode + 1) % print_freq == 0:
            # Recent averages are only progress prints Tests decide saves
            sample_count = min(len(recent_wins), print_freq)
            win_rate = sum(recent_wins[-sample_count:]) / sample_count * 100.0
            avg_reward = sum(recent_rewards[-sample_count:]) / sample_count
            avg_pins = sum(recent_pins[-sample_count:]) / sample_count
            elapsed = time.time() - start_time
            opponent_tag = opponent_label if opponent_label.startswith("Hist:") else opponent_label.capitalize()
            print(
                f"  ep {episode + 1:4d}/{CONTINUE_EPISODES}   [{opponent_tag:7s}]   "
                f"e={agent.epsilon:.3f}   "
                f"steps={steps:3d}   wins={win_rate:4.1f}%   avgR={avg_reward:+.2f}   "
                f"pins={avg_pins:.1f}/10   t={elapsed:.0f}s"
            )
        if (episode + 1) % EVAL_FREQ == 0:
            # Test each model before it can become the best one
            wr_g, pins_g = evaluate(agent, greedy_opp, games=EVAL_GAMES, max_turns=MAX_TURNS)
            wr_e, pins_e = evaluate(agent, easy_opp, games=EVAL_GAMES, max_turns=500)
            pair_eval = evaluate_afterstate_by_pair(
                agent,
                greedy_opp,
                games_per_pair=PAIR_EVAL_GAMES,
                max_turns=MAX_TURNS,
            )
            weakest_lane = weakest_afterstate_pair_summary(pair_eval["pairs"])
            print(f"       eval Greedy      wins={wr_g:4.1f}%   pins={pins_g:.1f}/10")
            print(f"       eval EasyRandom  wins={wr_e:4.1f}%   pins={pins_e:.1f}/10")
            print(
                f"       weakest lane    {weakest_lane['pair']:<17s} "
                f"score={weakest_lane['score']:.1f}   wins={weakest_lane['win_rate']:.1f}%   "
                f"pins={weakest_lane['avg_pins']:.1f}/10"
            )

            greedy_score = model_selection_score(wr_g, pins_g)
            easy_score = model_selection_score(wr_e, pins_e)
            # Greedy matters most because it is the harder opponent here
            eval_score = greedy_score + 0.15 * easy_score
            hard_collapse = (
                wr_g <= max(0.0, baseline_greedy_wins - hard_collapse_win_margin)
                or pins_g <= max(0.0, baseline_greedy_pins - hard_collapse_pin_margin)
                or eval_score <= baseline_eval_score - collapse_stop_margin
            )

            # Stop early if training clearly hurts the loaded model
            if hard_collapse:
                print(
                    f"       early stop   hard collapse below baseline   "
                    f"(Greedy {wr_g:.1f}%/{pins_g:.1f}, baseline {baseline_greedy_wins:.1f}%/{baseline_greedy_pins:.1f})"
                )
                break

            if eval_score > best_eval_score:
                weakest_floor = best_weakest_score - WEAKEST_PAIR_MARGIN
                if eval_score > best_external_score:
                    # Save a backup if it looks good, even if it is not final
                    best_external_score = eval_score
                    agent.save_model(CHALLENGER_EXTERNAL_MODEL)
                    print("       *** New external-best challenger saved ***")
                if weakest_lane["score"] >= weakest_floor:
                    # Only save as best if it improves and both lanes still work
                    best_eval_score = eval_score
                    best_eval_episode = episode + 1
                    best_weakest_score = weakest_lane["score"]
                    best_weakest_pair = weakest_lane["pair"]
                    evals_without_improvement = 0
                    current_best_candidate_path = CHALLENGER_BEST_MODEL
                    current_best_candidate_is_challenger = True
                    agent.save_model(CHALLENGER_BEST_MODEL)
                    agent.save_model(CHALLENGER_TRAINED_MODEL)
                    print("       *** New best model saved ***")
                else:
                    evals_without_improvement += 1
                    print(
                        f"       promotion blocked: weakest lane {weakest_lane['pair']} "
                        f"fell below floor {weakest_floor:.1f}"
                    )
            else:
                evals_without_improvement += 1
                collapsed = best_eval_score > float("-inf") and eval_score <= best_eval_score - collapse_stop_margin
                # Stop after several tests without improvement
                if ((episode + 1) >= min_episodes_before_early_stop and
                        (evals_without_improvement >= early_stop_patience or collapsed)):
                    reason = "collapsed below best" if collapsed else "no eval improvement"
                    print(
                        f"       early stop   {reason} for {evals_without_improvement} checkpoints"
                        f"   (best at ep {best_eval_episode})"
                    )
                    break

    print("\n\nAfterstate Continuation Complete\n")
    print(f"  Time      {time.time() - start_time:.0f}s")
    print(f"  Epsilon   {agent.epsilon:.4f}")
    print(f"  Memory    online {len(agent.memory)}   demo {len(agent.demo_memory)}")
    if best_eval_episode:
        print(f"  Best eval at ep {best_eval_episode}")
    else:
        print("  Best eval remained at loaded checkpoint")
    print(f"  Best weakest lane   {best_weakest_pair}   score {best_weakest_score:.1f}")
    print(f"  Best external score   {best_external_score:.1f}")
    print(f"  Challenger best   {CHALLENGER_BEST_MODEL}")

    print(f"\nFinal test vs Greedy   {EVAL_GAMES} games\n")
    candidate_path = current_best_candidate_path
    # Final tests use the best saved model, not the last model in memory
    agent.load_model(candidate_path)
    agent.epsilon = 0.0
    final_wins, final_pins = evaluate(agent, greedy_opp, games=EVAL_GAMES, max_turns=MAX_TURNS)
    final_pair_eval = evaluate_afterstate_by_pair(
        agent,
        greedy_opp,
        games_per_pair=PAIR_EVAL_GAMES,
        max_turns=MAX_TURNS,
    )
    final_weakest = weakest_afterstate_pair_summary(final_pair_eval["pairs"])
    print(f"  Result   {final_wins:.1f}% wins   pins={final_pins:.1f}/10")
    print(
        f"  Weakest lane   {final_weakest['pair']}   score {final_weakest['score']:.1f}   "
        f"wins={final_weakest['win_rate']:.1f}%   pins={final_weakest['avg_pins']:.1f}/10"
    )
    print(f"\nFinal test vs Minimax(depth={MINIMAX_DEPTH})   {EVAL_GAMES} games\n")
    minimax_wins, minimax_pins = evaluate(agent, minimax_opp, games=EVAL_GAMES, max_turns=MAX_TURNS)
    print(f"  Result   {minimax_wins:.1f}% wins   pins={minimax_pins:.1f}/10")

    promotion_floor = best_weakest_score - WEAKEST_PAIR_MARGIN
    promotion_reasons = []
    # Final tests are stricter because this can replace the public model
    if final_wins < 70.0:
        promotion_reasons.append(f"Greedy wins {final_wins:.1f}% < 70.0%")
    if final_pins < 9.0:
        promotion_reasons.append(f"Greedy pins {final_pins:.1f} < 9.0")
    if minimax_wins < 50.0:
        promotion_reasons.append(f"Minimax wins {minimax_wins:.1f}% < 50.0%")
    if final_weakest["score"] < promotion_floor:
        promotion_reasons.append(
            f"weakest lane {final_weakest['pair']} score {final_weakest['score']:.1f} < {promotion_floor:.1f}"
        )

    should_promote = promotion_enabled and not promotion_reasons
    if should_promote:
        # Only now copy into the final files used by the tournament player
        copy_if_different(candidate_path, AFTERSTATE_TWOPLAYER_MODEL)
        copy_if_different(candidate_path, AFTERSTATE_TRAINED_MODEL)
        if os.path.exists(CHALLENGER_EXTERNAL_MODEL):
            copy_if_different(CHALLENGER_EXTERNAL_MODEL, AFTERSTATE_EXTERNAL_MODEL)
        if current_best_candidate_is_challenger:
            print(f"\nPromoted challenger to tournament afterstate model: {AFTERSTATE_TWOPLAYER_MODEL}")
        else:
            print(f"\nKept loaded checkpoint as tournament afterstate model: {AFTERSTATE_TWOPLAYER_MODEL}")
    elif not promotion_enabled:
        print("\nPromotion disabled by --no-promote; kept existing tournament afterstate model.")
        if promotion_reasons:
            print("Candidate would also have been blocked:")
            for reason in promotion_reasons:
                print(f"  blocked: {reason}")
    else:
        print("\nKept existing tournament afterstate model; challenger did not pass promotion gate.")
        for reason in promotion_reasons:
            print(f"  blocked: {reason}")
        if os.path.exists(CHALLENGER_EXTERNAL_MODEL):
            copy_if_different(CHALLENGER_EXTERNAL_MODEL, AFTERSTATE_EXTERNAL_MODEL)

if __name__ == "__main__":
    main()

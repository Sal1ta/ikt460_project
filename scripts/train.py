# Train the Afterstate agent from scratch

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.afterstate import AfterstateValueAgent
from src.agents import GreedyAgent, HomeFirstRandomAgent
from src.env import ChineseCheckersEnv
from src.paths import (
    AFTERSTATE_TWOPLAYER_MODEL,
    AFTERSTATE_CHECKPOINT_DIR,
    AFTERSTATE_EXTERNAL_MODEL,
    AFTERSTATE_FINAL_MODEL,
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

PHASE1_EPISODES = 300
PHASE2_EPISODES = 900
TOTAL_EPISODES = PHASE1_EPISODES + PHASE2_EPISODES

# Main training and testing settings
WARMUP_EPISODES = 200
MAX_TURNS = 400
EVAL_FREQ = 100
EVAL_GAMES = 10
PAIR_EVAL_GAMES = 10
CHECKPOINT_FREQ = 500
TARGET_UPDATE_STEPS  = 250
WEAKEST_PAIR_MARGIN = 20.0

# Save rules make sure the model works against Greedy too
PROMOTION_MIN_GREEDY_WINS = 35.0
PROMOTION_MIN_GREEDY_PINS = 7.0
PROMOTION_MIN_LANE_SCORE = 100.0
# Do several CPU moves between replay updates
REPLAY_FREQUENCY     = 4

BEST_MODEL = str(AFTERSTATE_TWOPLAYER_MODEL)
EXTERNAL_MODEL = str(AFTERSTATE_EXTERNAL_MODEL)
FINAL_MODEL = str(AFTERSTATE_FINAL_MODEL)
COMPAT_MODEL = str(AFTERSTATE_TRAINED_MODEL)

shaped_reward = afterstate_shaped_reward
evaluate = evaluate_afterstate

def greedy_warmup(agent, episodes=WARMUP_EPISODES):
    # Start with teacher moves before normal training begins
    # This fills memory with legal forward moves
    print(f"\nGreedy warm-start   {episodes} games")
    teacher = GreedyAgent(name="WarmupTeacher")
    opponent = HomeFirstRandomAgent(name="WarmupEasyOpp")
    wins = 0

    for episode in range(episodes):
        # Change colour pair so the model learns both board directions
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
                # Let the opponent answer before saving the next state
                action = opponent.choose_action(env, valid_actions)
                if action is None:
                    break
                _, _, done, info = env.step(action)
                steps += 1
                continue

            action = teacher.choose_action(env, valid_actions)
            if action is None:
                break

            # Measure board value before and after the move
            phi_s = compute_potential(env, agent_color)
            progress = env.evaluate_action_progress(action, agent_color)
            afterstate = agent.afterstate_from_env_action(env, action, agent_color)
            _, _, done, info = env.step(action)
            steps += 1

            if not done:
                opponent_actions = env.get_valid_actions()
                if opponent_actions:
                    opponent_action = opponent.choose_action(env, opponent_actions)
                    if opponent_action is not None:
                        _, _, done, info = env.step(opponent_action)
                        steps += 1

            phi_sp = compute_potential(env, agent_color)
            reward = shaped_reward(done, info, agent_color, phi_s, phi_sp, progress)
            next_position_state = agent.position_state_from_env(env, agent_color) if not done else None
            # Mark teacher examples so replay can sample some of them
            agent.remember(afterstate, reward, next_position_state, done, demo=True)

        if f"{agent_color} wins" in info.get("message", ""):
            wins += 1

        agent.finish_episode()

        if (episode + 1) % 20 == 0:
            print(f"  {episode + 1:3d}/{episodes}   demo {len(agent.demo_memory)}   wins {wins}/{episode + 1}")

    pretrain_steps = min(250, len(agent.demo_memory) // max(1, agent.batch_size))
    for _ in range(pretrain_steps):
        # Train on teacher examples before normal play starts
        agent.replay()
    agent.update_target_network()
    print(f"  Done   {len(agent.demo_memory)} experiences   {wins}/{episodes} wins   {pretrain_steps} pretrain steps\n")

def main():
    # Optional number makes the run shorter
    requested = int(sys.argv[1]) if len(sys.argv) > 1 else None
    phase1 = min(PHASE1_EPISODES, requested) if requested else PHASE1_EPISODES
    phase2 = max(0, (requested or TOTAL_EPISODES) - phase1)
    episodes = phase1 + phase2

    # Ask the environment how large the model input should be
    probe_env = ChineseCheckersEnv(num_players=2, player_colors=["yellow", "purple"])
    state_size = len(probe_env.reset())

    agent = AfterstateValueAgent(state_size=state_size, player_color="yellow", name="AfterstateAgent")
    agent.demo_fraction = 0.50

    easy_opponent = HomeFirstRandomAgent(name="EasyOpp")
    greedy_opponent = GreedyAgent(name="GreedyOpp")

    print("\nChinese Checkers   Afterstate Training")
    print()
    greedy_warmup(agent, episodes=WARMUP_EPISODES)
    print("Training Setup")
    print()
    print(f"  Episodes   {episodes}   ({phase1} vs EasyRandom   {phase2} vs Greedy)")
    print(f"  State      {state_size} features")
    print(f"  Color lanes {', '.join(f'{a}/{b}' for a, b in PLAYABLE_COLOR_PAIRS)}")
    print(f"  Epsilon    {agent.epsilon:.2f} -> {agent.epsilon_min:.2f}   decay {agent.epsilon_decay}")
    print("  Reward     win +100   timeout/draw -30   loss -50   + clipped progress + PBRS")
    print(f"  Replay     online {agent.memory.maxlen}   demo {agent.demo_memory.maxlen}   demo frac {agent.demo_fraction:.2f}")
    print(f"  Targets    {agent.n_step}-step bootstrap")
    print(f"  Target     sync every {TARGET_UPDATE_STEPS} replay updates")
    print(f"  Eval       every {EVAL_FREQ} episodes with epsilon=0")
    print(f"  Pair eval   Greedy by lane   {PAIR_EVAL_GAMES} games per pair")
    print(f"  Lane floor  best weakest lane - {WEAKEST_PAIR_MARGIN:.1f} score")
    print(
        f"  Promote     Greedy >= {PROMOTION_MIN_GREEDY_WINS:.0f}%   "
        f"pins >= {PROMOTION_MIN_GREEDY_PINS:.1f}   lane >= {PROMOTION_MIN_LANE_SCORE:.1f}"
    )
    print()
    print("Training")
    print()

    best_score = float("-inf")
    best_episode = -1
    best_weakest_score = float("-inf")
    best_weakest_pair = "n/a"
    best_external_score = float("-inf")
    no_improve = 0
    # Keep these for terminal summaries after a long run
    phase1_best_score = float("-inf")
    phase2_best_score = float("-inf")
    start_time = time.time()

    # Do not replace a good saved model unless this one is clearly better
    if AFTERSTATE_TWOPLAYER_MODEL.exists():
        baseline_agent = AfterstateValueAgent(state_size=state_size, player_color="yellow", name="AfterstateBaseline")
        if baseline_agent.load_model(BEST_MODEL, verbose=False):
            baseline_easy_wins, baseline_easy_pins = evaluate(
                baseline_agent,
                easy_opponent,
                games=EVAL_GAMES,
                max_turns=MAX_TURNS,
            )
            baseline_greedy_wins, baseline_greedy_pins = evaluate(
                baseline_agent,
                greedy_opponent,
                games=EVAL_GAMES,
                max_turns=MAX_TURNS,
            )
            baseline_pair_eval = evaluate_afterstate_by_pair(
                baseline_agent,
                greedy_opponent,
                games_per_pair=PAIR_EVAL_GAMES,
                max_turns=MAX_TURNS,
            )
            baseline_weakest_lane = weakest_afterstate_pair_summary(baseline_pair_eval["pairs"])
            baseline_easy_score = model_selection_score(baseline_easy_wins, baseline_easy_pins)
            baseline_greedy_score = model_selection_score(baseline_greedy_wins, baseline_greedy_pins)
            # The starting score becomes the first score to beat
            best_score = baseline_greedy_score + 0.10 * baseline_easy_score
            best_external_score = best_score
            best_weakest_score = baseline_weakest_lane["score"]
            best_weakest_pair = baseline_weakest_lane["pair"]
            best_episode = 0
            print(
                f"  Existing best baseline   Greedy {baseline_greedy_wins:.1f}%   "
                f"pins {baseline_greedy_pins:.1f}   score {best_score:.1f}"
            )
            print(
                f"  Baseline weakest lane   {best_weakest_pair}   "
                f"score {best_weakest_score:.1f}"
            )
            print()

    recent_wins = []
    recent_pins = []
    recent_rewards = []
    print_frequency = max(1, min(20, episodes // 25))

    for episode in range(episodes):
        # Phase 1 uses an easier opponent Phase 2 switches to Greedy
        in_phase1 = episode < phase1
        opponent = easy_opponent if in_phase1 else greedy_opponent
        opp_tag = "EasyRandom" if in_phase1 else "Greedy"
        player_colors = color_pair_for_game(episode)
        agent_color = player_colors[0]
        env = ChineseCheckersEnv(num_players=2, player_colors=player_colors, max_turns=MAX_TURNS)

        if episode == phase1 and phase2 > 0:
            # Phase 2 adds a bit more exploration without resetting the model
            agent.epsilon = max(agent.epsilon, 0.10)
            no_improve = 0
            print(f"\n--- Phase 2 start (ep {episode})  ε set to {agent.epsilon:.2f} ---\n")

        env.reset()
        done = False
        info = {}
        steps = 0
        total_reward = 0.0
        won = False

        while not done and steps < MAX_TURNS:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            if env.get_current_player() == agent_color:
                # Save the board after our move as the training example
                action = agent.choose_action(env, valid_actions)
                if action is None:
                    break

                phi_s = compute_potential(env, agent_color)
                progress = env.evaluate_action_progress(action, agent_color)
                afterstate = agent.afterstate_from_env_action(env, action, agent_color)
                _, _, done, info = env.step(action)
                steps += 1

                if not done:
                    # Let the opponent reply so the next state is our turn again
                    opponent_actions = env.get_valid_actions()
                    if opponent_actions:
                        opponent_action = opponent.choose_action(env, opponent_actions)
                        if opponent_action is not None:
                            _, _, done, info = env.step(opponent_action)
                            steps += 1

                phi_sp = compute_potential(env, agent_color)
                reward = shaped_reward(done, info, agent_color, phi_s, phi_sp, progress)
                next_position_state = agent.position_state_from_env(env, agent_color) if not done else None
                # The model learns to value the board after a move
                agent.remember(afterstate, reward, next_position_state, done)
                total_reward += reward

                if steps % REPLAY_FREQUENCY == 0:
                    # Update the target network after enough replay updates
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
            # Reduce exploration once per game
            agent.epsilon *= agent.epsilon_decay

        agent.finish_episode()

        recent_wins.append(1 if won else 0)
        recent_pins.append(env.count_player_pins_in_target(agent_color))
        recent_rewards.append(total_reward)

        if (episode + 1) % CHECKPOINT_FREQ == 0:
            agent.save_model(str(AFTERSTATE_CHECKPOINT_DIR / f"phase1_ep{episode + 1}.pth"))

        if (episode + 1) % print_frequency == 0:
            # Recent scores are noisy, but they help spot bad drops
            sample_count = min(len(recent_wins), print_frequency)
            window_wins = recent_wins[-sample_count:]
            window_pins = recent_pins[-sample_count:]
            window_rewards = recent_rewards[-sample_count:]
            win_rate = sum(window_wins) / sample_count * 100.0
            avg_pins = sum(window_pins) / sample_count
            avg_reward = sum(window_rewards) / sample_count
            print(
                f"  ep {episode + 1:4d}/{episodes}  [{opp_tag}]   "
                f"e={agent.epsilon:.3f}   steps={steps:3d}   wins={win_rate:4.1f}%   "
                f"avgR={avg_reward:+.2f}   pins={avg_pins:.1f}/10   t={time.time() - start_time:.0f}s"
            )

        if (episode + 1) % EVAL_FREQ == 0:
            # Testing uses the same scoring helper as the report
            easy_wins, easy_pins = evaluate(agent, easy_opponent, games=EVAL_GAMES, max_turns=MAX_TURNS)
            greedy_wins, greedy_pins = evaluate(agent, greedy_opponent, games=EVAL_GAMES, max_turns=MAX_TURNS)
            pair_eval = evaluate_afterstate_by_pair(
                agent,
                greedy_opponent,
                games_per_pair=PAIR_EVAL_GAMES,
                max_turns=MAX_TURNS,
            )
            weakest_lane = weakest_afterstate_pair_summary(pair_eval["pairs"])
            print(f"       eval EasyRandom   wins={easy_wins:.1f}%   pins={easy_pins:.1f}/10")
            print(f"       eval Greedy      wins={greedy_wins:.1f}%   pins={greedy_pins:.1f}/10")
            print(
                f"       weakest lane    {weakest_lane['pair']:<17s} "
                f"score={weakest_lane['score']:.1f}   wins={weakest_lane['win_rate']:.1f}%   "
                f"pins={weakest_lane['avg_pins']:.1f}/10"
            )

            easy_score = model_selection_score(easy_wins, easy_pins)
            greedy_score = model_selection_score(greedy_wins, greedy_pins)

            if in_phase1:
                eval_score = easy_score + 0.10 * greedy_score
            else:
                # Phase 2 mostly cares about Greedy, but still checks EasyRandom
                eval_score = greedy_score + 0.10 * easy_score

            if in_phase1:
                phase1_best_score = max(phase1_best_score, eval_score)
            else:
                phase2_best_score = max(phase2_best_score, eval_score)

            greedy_gate_ok = (
                greedy_wins >= PROMOTION_MIN_GREEDY_WINS
                and greedy_pins >= PROMOTION_MIN_GREEDY_PINS
                and weakest_lane["score"] >= PROMOTION_MIN_LANE_SCORE
            )

            # Backup and best files are separate from the tournament model
            if greedy_gate_ok and eval_score > best_external_score:
                best_external_score = eval_score
                agent.save_model(EXTERNAL_MODEL)
                print("       *** New external-best model saved ***")
            elif eval_score > best_external_score:
                print("       external-best blocked: Greedy gate not reached yet")

            if eval_score > best_score:
                weakest_floor = (
                    float("-inf")
                    if best_weakest_score == float("-inf")
                    else best_weakest_score - WEAKEST_PAIR_MARGIN
                )
                if not greedy_gate_ok:
                    no_improve += 1
                    print(
                        "       promotion blocked: Greedy gate not reached "
                        f"({greedy_wins:.1f}% / {greedy_pins:.1f} pins / lane {weakest_lane['score']:.1f})"
                    )
                elif weakest_lane["score"] >= weakest_floor:
                    # Save only if the score improves and both colour lanes still work
                    best_score = eval_score
                    best_episode = episode + 1
                    best_weakest_score = weakest_lane["score"]
                    best_weakest_pair = weakest_lane["pair"]
                    no_improve = 0
                    agent.save_model(BEST_MODEL)
                    agent.save_model(COMPAT_MODEL)
                    print("       *** New best model saved ***")
                else:
                    no_improve += 1
                    print(
                        f"       promotion blocked: weakest lane {weakest_lane['pair']} "
                        f"fell below floor {weakest_floor:.1f}"
                    )
            else:
                no_improve += 1
                if (not in_phase1) and no_improve >= 8:
                    print(
                        f"       early stop   no eval improvement for {no_improve} checkpoints   "
                        f"(best at ep {best_episode})"
                    )
                    break

    elapsed = time.time() - start_time
    print()
    print("Training Complete")
    print()
    print(f"  Time      {elapsed:.0f}s")
    print(f"  Epsilon   {agent.epsilon:.4f}")
    print(f"  Wins      {sum(recent_wins[-200:]) / max(1, len(recent_wins[-200:])) * 100:.1f}%   (last 200 eps)")
    print(f"  Pins      {sum(recent_pins[-200:]) / max(1, len(recent_pins[-200:])):.1f}/10")
    print(f"  Memory    online {len(agent.memory)}   demo {len(agent.demo_memory)}")
    if best_episode > 0:
        print(f"  Best eval at ep {best_episode}")
        print(f"  Best weakest lane   {best_weakest_pair}   score {best_weakest_score:.1f}")
    elif best_episode == 0:
        print("  Best eval remained at the existing checkpoint")
        print(f"  Best weakest lane   {best_weakest_pair}   score {best_weakest_score:.1f}")
    if best_external_score > float("-inf"):
        print(f"  Best external score   {best_external_score:.1f}")

    if best_episode >= 0:
        # Also save the old filename for code that still expects it
        agent.load_model(BEST_MODEL)
        agent.save_model(FINAL_MODEL)
        agent.save_model(COMPAT_MODEL)
        print()
        print(f"  Best model saved to {COMPAT_MODEL}")
    else:
        agent.save_model(FINAL_MODEL)

    print()
    print(f"Test vs Greedy   {EVAL_GAMES} games")
    print()
    final_wins, final_pins = evaluate(agent, greedy_opponent, games=EVAL_GAMES, max_turns=MAX_TURNS)
    final_pair_eval = evaluate_afterstate_by_pair(
        agent,
        greedy_opponent,
        games_per_pair=PAIR_EVAL_GAMES,
        max_turns=MAX_TURNS,
    )
    final_weakest = weakest_afterstate_pair_summary(final_pair_eval["pairs"])
    print(f"  Result   {final_wins:.0f}% wins   avg pins {final_pins:.1f}/10")
    print(
        f"  Weakest lane   {final_weakest['pair']}   score {final_weakest['score']:.1f}   "
        f"wins {final_weakest['win_rate']:.1f}%   pins {final_weakest['avg_pins']:.1f}/10"
    )

if __name__ == "__main__":
    main()

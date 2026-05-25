# Extra training for the multiplayer Afterstate agent

import argparse
import contextlib
import csv
import io
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.afterstate import AfterstateValueAgent
from src.agents import GreedyAgent, RandomAgent
from src.env import ChineseCheckersEnv
from src.paths import AFTERSTATE_TWOPLAYER_MODEL, AFTERSTATE_MODEL_DIR, ensure_project_dirs
from src.rewards import afterstate_shaped_reward, compute_potential
from scripts.benchmark import SUPPORTED_OPPONENT_KINDS, build_frozen_opponent


# The latest model is saved at each test
# The best model is used by the tournament player
DEFAULT_OUTPUT = AFTERSTATE_MODEL_DIR / "multiplayerlatest.pth"
DEFAULT_BEST_OUTPUT = AFTERSTATE_MODEL_DIR / "multiplayer.pth"
DEFAULT_HISTORY_CSV = AFTERSTATE_MODEL_DIR / "traininghistory.csv"


def quiet_call(fn, *args, **kwargs):
    # Hide extra print text during long training runs
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return fn(*args, **kwargs)


def notify_finished(message):
    # Mac shows a notification Other systems just print the result
    if platform.system() != "Darwin":
        print(f"\n{message}")
        return
    script = f'display notification "{message}" with title "Chinese Checkers" sound name "Glass"'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print(f"\n{message}")


def parse_players(raw):
    # Do not train 2 player here, so the 2 player model is not damaged
    players = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        # This multiplayer model is used for 3, 4, 5, and 6 players
        if value not in (3, 4, 5, 6):
            raise ValueError("players must be chosen from 3,4,5,6")
        players.append(value)
    if not players:
        raise ValueError("at least one player count is required")
    return players


def parse_mix(raw):
    # Use the same opponent names as the benchmark script
    mix = [token.strip().lower() for token in str(raw).split(",") if token.strip()]
    valid = set(SUPPORTED_OPPONENT_KINDS)
    if not mix or any(item not in valid for item in mix):
        raise ValueError(
            "opponent mix entries must be chosen from "
            + ", ".join(sorted(valid))
        )
    return mix


def make_agent(source_model):
    # The new model starts from the 2 player model but is saved separately
    probe_env = ChineseCheckersEnv(num_players=2, player_colors=["yellow", "purple"])
    quiet_call(probe_env.reset)
    agent = AfterstateValueAgent(
        state_size=len(probe_env.get_state_for_player("yellow")),
        player_color="yellow",
        name="MultiplayerChallenger",
    )
    if not agent.load_model(str(source_model)):
        raise RuntimeError(f"Could not load source model: {source_model}")
    return agent


def make_opponents(kind, player_colors, agent_colour, frozen_pool=None):
    # Loaded opponents are reused Simple opponents are made when needed
    frozen_pool = frozen_pool or {}
    if kind in frozen_pool:
        frozen = frozen_pool[kind]
        return {
            # Other colours share this fixed learned opponent
            colour: frozen
            for colour in player_colors
            if colour != agent_colour
        }
    cls = GreedyAgent if kind == "greedy" else RandomAgent
    return {
        # Greedy and random agents are simple enough to make per colour
        colour: cls(name=f"{kind}-{colour}")
        for colour in player_colors
        if colour != agent_colour
    }


def run_training_episode(agent, num_players, opponent_kind, agent_seat, max_turns, replay_frequency, frozen_pool=None):
    # Change the controlled seat so the model sees every colour
    env = ChineseCheckersEnv(num_players=num_players, max_turns=max_turns)
    quiet_call(env.reset)
    player_colors = list(env.player_colors)
    # Keep the seat inside the number of players
    agent_colour = player_colors[agent_seat % num_players]
    opponents = make_opponents(opponent_kind, player_colors, agent_colour, frozen_pool=frozen_pool)

    done = False
    info = {}
    steps = 0
    own_moves = 0
    total_reward = 0.0

    while not done and steps < max_turns:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break

        current = env.get_current_player()
        if current != agent_colour:
            # Let fixed opponents play until it is our turn
            action = opponents[current].choose_action(env, valid_actions)
            if action is None:
                break
            _, _, done, info = env.step(action)
            steps += 1
            continue

        action = agent.choose_action(env, valid_actions, explore=True)
        if action is None:
            break

        # Save values before making the move
        phi_s = compute_potential(env, agent_colour)
        progress = env.evaluate_action_progress(action, agent_colour)
        afterstate = agent.afterstate_from_env_action(env, action, agent_colour)
        _, _, done, info = env.step(action)
        steps += 1
        own_moves += 1

        # Store our move and the opponent replies after it
        while not done and steps < max_turns and env.get_current_player() != agent_colour:
            opponent_actions = env.get_valid_actions()
            if not opponent_actions:
                break
            opponent = opponents[env.get_current_player()]
            opponent_action = opponent.choose_action(env, opponent_actions)
            if opponent_action is None:
                break
            _, _, done, info = env.step(opponent_action)
            steps += 1

        phi_sp = compute_potential(env, agent_colour)
        reward = afterstate_shaped_reward(done, info, agent_colour, phi_s, phi_sp, progress)
        next_position_state = agent.position_state_from_env(env, agent_colour) if not done else None
        # Save one training example for each move our agent makes
        agent.remember(afterstate, reward, next_position_state, done)
        total_reward += reward

        if own_moves % replay_frequency == 0:
            # Replay only sometimes, because multiplayer games are long
            if agent.replay() and agent.train_updates % 250 == 0:
                agent.update_target_network()

    agent.finish_episode()
    # One final replay helps short games still learn
    if agent.replay() and agent.train_updates % 250 == 0:
        agent.update_target_network()

    won = f"{agent_colour} wins" in info.get("message", "")
    return {
        "won": won,
        "pins": env.count_player_pins_in_target(agent_colour),
        "distance": env.total_distance_to_target(agent_colour),
        "reward": total_reward,
        "steps": steps,
        "own_moves": own_moves,
        "agent_colour": agent_colour,
        "result": info.get("message", "Unknown"),
    }


def run_eval_match(agent, num_players, opponent_kind, agent_seat, max_turns, frozen_pool=None):
    # Turn off exploration during testing, then restore it
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    env = ChineseCheckersEnv(num_players=num_players, max_turns=max_turns)
    quiet_call(env.reset)
    player_colors = list(env.player_colors)
    agent_colour = player_colors[agent_seat % num_players]
    opponents = make_opponents(opponent_kind, player_colors, agent_colour, frozen_pool=frozen_pool)

    done = False
    info = {}
    steps = 0
    while not done and steps < max_turns:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        current = env.get_current_player()
        if current == agent_colour:
            # No random moves during testing
            action = agent.choose_action(env, valid_actions, explore=False)
        else:
            action = opponents[current].choose_action(env, valid_actions)
        if action is None:
            break
        _, _, done, info = env.step(action)
        steps += 1

    scoreboard = []
    for colour in player_colors:
        # Use the same simple score parts as local benchmarks
        pins = env.count_player_pins_in_target(colour)
        distance = env.total_distance_to_target(colour)
        won = info.get("message") == f"{colour} wins"
        score = pins * 100.0 + max(0.0, 400.0 - 2.0 * distance) + (1000.0 if won else 0.0)
        scoreboard.append((score, colour, pins, distance, won))
    scoreboard.sort(reverse=True)
    # Rank uses the full scoreboard, not only win or loss
    rank = next(index for index, (_, colour, *_rest) in enumerate(scoreboard) if colour == agent_colour) + 1
    own = next(item for item in scoreboard if item[1] == agent_colour)

    agent.epsilon = old_epsilon
    return {
        "won": own[4],
        "rank": rank,
        "pins": own[2],
        "distance": own[3],
        "score": own[0],
    }


def evaluate_multiplayer(agent, players, games, max_turns, opponent_kind="greedy", frozen_pool=None):
    # Test different seats so one bad colour does not dominate
    summary = {}
    for num_players in players:
        results = [
            # Use % so the seat always exists
            run_eval_match(agent, num_players, opponent_kind, seat % num_players, max_turns, frozen_pool=frozen_pool)
            for seat in range(games)
        ]
        wins = sum(1 for result in results if result["won"])
        avg_rank = sum(result["rank"] for result in results) / games
        avg_pins = sum(result["pins"] for result in results) / games
        avg_score = sum(result["score"] for result in results) / games
        summary[num_players] = {
            "wins": wins,
            "win_rate": wins / games * 100.0,
            "avg_rank": avg_rank,
            "avg_pins": avg_pins,
            "avg_score": avg_score,
        }
    return summary


def print_eval(summary, games):
    for num_players, stats in summary.items():
        print(
            f"       eval {num_players}p/{games}g   wins={stats['wins']:2d} "
            f"({stats['win_rate']:4.1f}%)   rank={stats['avg_rank']:.2f}   "
            f"pins={stats['avg_pins']:.2f}   score={stats['avg_score']:.1f}"
        )


def combined_eval_score(summary):
    # Lower rank is better This score also rewards wins and pins
    total = 0.0
    for num_players, stats in summary.items():
        # Reward ranks that beat a normal random seat result
        neutral_rank = (num_players + 1) / 2.0
        total += stats["win_rate"] + stats["avg_pins"] * 8.0 + (neutral_rank - stats["avg_rank"]) * 15.0
    return total


def append_history_rows(path, episode, phase, summary, eval_score, promoted):
    # Keep the CSV small one row per player count at each test
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        fieldnames = [
            "episode",
            "phase",
            "players",
            "wins",
            "win_rate",
            "avg_rank",
            "avg_pins",
            "avg_score",
            "combined_score",
            "promoted",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            # Append rows so long training can continue later
            writer.writeheader()
        for num_players, stats in summary.items():
            writer.writerow({
                "episode": episode,
                "phase": phase,
                "players": num_players,
                "wins": stats["wins"],
                "win_rate": round(stats["win_rate"], 4),
                "avg_rank": round(stats["avg_rank"], 4),
                "avg_pins": round(stats["avg_pins"], 4),
                "avg_score": round(stats["avg_score"], 4),
                "combined_score": round(eval_score, 4),
                "promoted": int(promoted),
            })


def main():
    # Defaults are short enough to finish locally, but longer runs are allowed
    parser = argparse.ArgumentParser(description="Train a separate multiplayer Afterstate model.")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--players", default="4,6", help="Comma-separated player counts from 3,4,6.")
    parser.add_argument(
        "--opponents",
        default="greedy",
        help=(
            "Training opponent mix from "
            + ", ".join(SUPPORTED_OPPONENT_KINDS)
            + "; comma-separated combinations allowed."
        ),
    )
    parser.add_argument("--eval-opponent", default="greedy", choices=SUPPORTED_OPPONENT_KINDS)
    parser.add_argument("--eval-games", type=int, default=12)
    parser.add_argument("--eval-freq", type=int, default=30)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--replay-frequency", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--source", default=str(AFTERSTATE_TWOPLAYER_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--best-output", default=str(DEFAULT_BEST_OUTPUT))
    parser.add_argument("--history-csv", default=str(DEFAULT_HISTORY_CSV))
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    if args.seed is not None:
        # Seed random choices so curves are easier to compare
        random.seed(args.seed)
        np.random.seed(args.seed)

    players = parse_players(args.players)
    opponent_mix = parse_mix(args.opponents)
    source = Path(args.source)
    output = Path(args.output)
    best_output = Path(args.best_output)
    history_csv = Path(args.history_csv)

    agent = make_agent(source)
    # Multiplayer still needs some random moves during training
    agent.epsilon = max(agent.epsilon, args.epsilon)
    agent.epsilon_min = min(agent.epsilon_min, 0.02)
    agent.epsilon_decay = 0.999
    frozen_pool = {}
    frozen_kinds = set(opponent_mix)
    frozen_kinds.add(args.eval_opponent)
    frozen_kinds.discard("greedy")
    frozen_kinds.discard("random")
    for kind in sorted(frozen_kinds):
        # Slow opponents are loaded once before training starts
        frozen_pool[kind] = build_frozen_opponent(kind)

    print("\nMultiplayer model training")
    print(f"  source:      {source}")
    print(f"  output:      {output}")
    print(f"  best output: {best_output}")
    print(f"  history csv: {history_csv}")
    print(f"  players:     {players}")
    print(f"  opponents:   {opponent_mix}")
    print(f"  eval opp.:   {args.eval_opponent}")
    print(f"  episodes:    {args.episodes}")
    print("  safety:      twoplayer.pth is never overwritten\n")

    start_time = time.time()
    print("  baseline before multiplayer fine-tuning:")
    baseline_summary = evaluate_multiplayer(
        agent,
        players=players,
        games=max(1, args.eval_games),
        max_turns=args.max_turns,
        opponent_kind=args.eval_opponent,
        frozen_pool=frozen_pool,
    )
    print_eval(baseline_summary, args.eval_games)
    best_score = combined_eval_score(baseline_summary)
    # Save the starting model as best first Later models must beat it
    agent.save_model(str(best_output))
    append_history_rows(history_csv, 0, "baseline", baseline_summary, best_score, True)
    print(f"       initial best multiplayer model: {best_output}\n")

    recent = []
    print_frequency = max(1, min(20, args.episodes // 10))

    for episode in range(args.episodes):
        # Cycle through player counts, opponents, and seats in a fixed order
        num_players = players[episode % len(players)]
        opponent_kind = opponent_mix[episode % len(opponent_mix)]
        agent_seat = episode % num_players
        result = run_training_episode(
            agent,
            num_players,
            opponent_kind,
            agent_seat,
            args.max_turns,
            args.replay_frequency,
            frozen_pool=frozen_pool,
        )
        recent.append(result)

        if agent.epsilon > agent.epsilon_min:
            # Reduce exploration after each game
            agent.epsilon *= agent.epsilon_decay

        if (episode + 1) % print_frequency == 0:
            window = recent[-print_frequency:]
            wins = sum(1 for item in window if item["won"])
            avg_pins = sum(item["pins"] for item in window) / len(window)
            avg_reward = sum(item["reward"] for item in window) / len(window)
            print(
                f"  ep {episode + 1:4d}/{args.episodes}   "
                f"{num_players}p/{opponent_kind:<6s}   e={agent.epsilon:.3f}   "
                f"wins={wins / len(window) * 100:4.1f}%   pins={avg_pins:.2f}   "
                f"avgR={avg_reward:+.2f}   updates={agent.train_updates}   "
                f"t={time.time() - start_time:.0f}s"
            )

        if (episode + 1) % max(1, args.eval_freq) == 0 or episode + 1 == args.episodes:
            # Save as best only when the combined score improves
            summary = evaluate_multiplayer(
                agent,
                players=players,
                games=max(1, args.eval_games),
                max_turns=args.max_turns,
                opponent_kind=args.eval_opponent,
                frozen_pool=frozen_pool,
            )
            print_eval(summary, args.eval_games)
            score = combined_eval_score(summary)
            agent.save_model(str(output))
            print(f"       saved multiplayer model: {output}")
            promoted = score > best_score
            if score > best_score:
                # Only the best tested model becomes multiplayerpth
                best_score = score
                agent.save_model(str(best_output))
                print(f"       *** new best multiplayer model: {best_output} ***")
            append_history_rows(history_csv, episode + 1, "eval", summary, score, promoted)

    agent.save_model(str(output))
    print(f"\nDone. Final multiplayer model saved to {output}")
    print(f"Best evaluated multiplayer model saved to {best_output}")
    print(f"Training/eval history saved to {history_csv}")
    print("\nEvaluate without replacing twoplayer.pth:")
    print(f"  python3 -u scripts/benchmark.py 4 greedy 36 tournament --model {best_output} --detail")
    print(f"  python3 -u scripts/benchmark.py 6 greedy 36 tournament --model {best_output} --detail")

    if args.notify:
        notify_finished("Multiplayer model training finished")


if __name__ == "__main__":
    main()

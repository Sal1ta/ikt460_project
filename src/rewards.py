# Reward and evaluation helpers for the learned agents

from src.env import ChineseCheckersEnv
from src.perspective import PLAYABLE_COLOR_PAIRS, color_pair_for_game

def compute_potential(env, player_color):
    # This packs pins already home and distance still left into one number
    # so shaped rewards can compare before and after a move
    pins = env.count_player_pins_in_target(player_color)
    dist = env.total_distance_to_target(player_color)
    return (pins * 10.0 - dist) / 14.0

def model_selection_score(win_rate_percent, avg_pins):
    # The report friendly score gives most of the weight to winning, while
    # still rewarding agents that consistently finish more pins
    return float(win_rate_percent) + float(avg_pins) * 10.0

def afterstate_shaped_reward(done, info, current_player, phi_s, phi_sp, action_progress):
    # Big outcomes dominate while a small progress bonus gives earlier feedback
    clipped_progress = max(-1.0, min(1.0, action_progress / 5.0))
    progress = 0.25 * clipped_progress

    if done:
        message = info.get("message", "").lower()
        if f"{current_player} wins" in message:
            return 100.0 + progress
        if any(token in message for token in ("max turns", "repetition", "draw")):
            return -30.0 + progress
        return -50.0 + progress

    # Keep the delta potential signal strong enough to influence mid game choices
    pbrs = 0.10 * (0.99 * phi_sp - phi_s)
    return progress + pbrs

def _run_afterstate_eval_game(agent, opponent, player_colors, max_turns=400, agent_first=True):
    # Pair aware evaluation needs the exact same move loop everywhere, so the
    # shared helper keeps those checks from drifting apart
    player_colors = list(player_colors)
    agent_color = player_colors[0] if agent_first else player_colors[1]
    env = ChineseCheckersEnv(num_players=2, player_colors=player_colors, max_turns=max_turns)
    env.reset()
    done = False
    info = {}
    steps = 0

    while not done and steps < max_turns:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        if env.get_current_player() == agent_color:
            action = agent.choose_action(env, valid_actions)
        else:
            action = opponent.choose_action(env, valid_actions)
        if action is None:
            break
        _, _, done, info = env.step(action)
        steps += 1

    return {
        "agent_color": agent_color,
        "win": f"{agent_color} wins" in info.get("message", ""),
        "pins": env.count_player_pins_in_target(agent_color),
        "steps": steps,
        "pair": tuple(player_colors),
    }

def evaluate_afterstate(agent, opponent, games=10, max_turns=400):
    # Evaluation runs with epsilon off to measure the learned policy directly
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    wins, pins = 0, []

    for game in range(games):
        result = _run_afterstate_eval_game(
            agent,
            opponent,
            color_pair_for_game(game),
            max_turns=max_turns,
            agent_first=True,
        )
        if result["win"]:
            wins += 1
        pins.append(result["pins"])

    agent.epsilon = old_epsilon
    return wins / games * 100.0, sum(pins) / games

def evaluate_afterstate_by_pair(agent, opponent, games_per_pair=4, max_turns=400):
    # Averages can hide one weak lane This keeps each opposite colour pair
    # visible during training and promotion
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    games_per_pair = max(2, int(games_per_pair))
    if games_per_pair % 2 == 1:
        games_per_pair += 1

    pair_results = {}
    total_wins = 0
    total_pins = 0.0
    total_steps = 0.0
    total_games = 0

    for pair_index, pair in enumerate(PLAYABLE_COLOR_PAIRS):
        pair_key = f"{pair[0]}/{pair[1]}"
        wins = 0
        pins = []
        steps = []

        for game in range(games_per_pair):
            result = _run_afterstate_eval_game(
                agent,
                opponent,
                pair,
                max_turns=max_turns,
                agent_first=(game % 2 == 0),
            )
            if result["win"]:
                wins += 1
                total_wins += 1
            pins.append(result["pins"])
            steps.append(result["steps"])
            total_pins += result["pins"]
            total_steps += result["steps"]
            total_games += 1

        pair_results[pair_key] = {
            "pair": tuple(pair),
            "games": games_per_pair,
            "win_rate": 100.0 * wins / games_per_pair,
            "avg_pins": sum(pins) / games_per_pair,
            "avg_steps": sum(steps) / games_per_pair,
            "pair_index": pair_index,
        }

    agent.epsilon = old_epsilon
    return {
        "overall_win_rate": 100.0 * total_wins / max(1, total_games),
        "overall_pins": total_pins / max(1, total_games),
        "overall_steps": total_steps / max(1, total_games),
        "pairs": pair_results,
        "games": total_games,
    }

def weakest_afterstate_pair_summary(pair_results):
    # Promotion should account for the weakest lane as well as the average lane
    if not pair_results:
        return {
            "pair": "n/a",
            "score": 0.0,
            "win_rate": 0.0,
            "avg_pins": 0.0,
            "avg_steps": 0.0,
        }

    scored_pairs = []
    for pair_key, pair_result in pair_results.items():
        pair_score = model_selection_score(pair_result["win_rate"], pair_result["avg_pins"])
        scored_pairs.append((pair_score, pair_key, pair_result))

    pair_score, pair_key, pair_result = min(scored_pairs, key=lambda item: item[0])
    return {
        "pair": pair_key,
        "score": float(pair_score),
        "win_rate": float(pair_result["win_rate"]),
        "avg_pins": float(pair_result["avg_pins"]),
        "avg_steps": float(pair_result["avg_steps"]),
    }

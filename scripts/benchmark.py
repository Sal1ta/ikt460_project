# Benchmark tournament routes
import contextlib
import io
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.afterstate import AfterstateSearchAgent, AfterstateValueAgent
from src.agents import GreedyAgent, RandomAgent
from src.env import ChineseCheckersEnv
from src.paths import (
    AFTERSTATE_TWOPLAYER_MODEL,
    AFTERSTATE_EXTERNAL_MODEL,
    AFTERSTATE_FINAL_MODEL,
    AFTERSTATE_MULTIPLAYER_MODEL,
    AFTERSTATE_TRAINED_MODEL,
    first_existing,
)
import player as tournament_player

RUN_NOTIFY = "--notify" in sys.argv
RUN_DETAIL = "--detail" in sys.argv
BOOLEAN_FLAGS = {"--detail", "--notify"}
VALUE_FLAGS = {"--seed", "--dense-gap", "--jump-weight", "--model"}
SUPPORTED_OPPONENT_KINDS = (
    # Opponent names used by benchmark and training
    "greedy",
    "random",
    "afterstate",
    "challenger",
    "original",
    "fallback",
    "tournament",
    "settle75",
    "seatbase",
    "seatmix",
    "corridor",
    "seatcorridor",
    "corridor4seatbase",
    "brs",
    "seatbrs",
    "external_afterstate",
)

def notify_finished(title="Chinese Checkers", message="Multiplayer evaluation finished"):
    # Show a Mac notification when a long run finishes
    if platform.system() != "Darwin":
        print(f"\n{message}")
        return

    script = (
        f'display notification "{message}" '
        f'with title "{title}" sound name "Glass"'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print(f"\n{message}")


def _positional_args(argv):
    # Keep quick commands short, while still allowing optional flags
    positional = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in VALUE_FLAGS:
            skip_next = True
            continue
        if any(arg.startswith(f"{flag}=") for flag in VALUE_FLAGS):
            continue
        if arg in BOOLEAN_FLAGS:
            continue
        if not arg.startswith("--"):
            positional.append(arg)
    return positional


def _flag_value(argv, flag_name, default=None):
    prefix = f"{flag_name}="
    for index, arg in enumerate(argv):
        if arg == flag_name and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def seed_everything(seed):
    # Use the same random choices each run
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def configure_afterstate(agent, dense_gap=None, jump_weight=None):
    # Optional tuning values for quick tests
    if dense_gap is not None:
        agent.dense_multiplayer_leader_gap = float(dense_gap)
    if jump_weight is not None:
        agent.dense_multiplayer_jump_weight = float(jump_weight)
    return agent


def load_afterstate(dense_gap=None, jump_weight=None, model_path=None, search_agent=False):
    # Ask the environment how large the model input should be
    path = (
        Path(model_path)
        if model_path is not None
        else first_existing(
            AFTERSTATE_TWOPLAYER_MODEL,
            AFTERSTATE_EXTERNAL_MODEL,
            AFTERSTATE_TRAINED_MODEL,
            AFTERSTATE_FINAL_MODEL,
        )
    )
    env = ChineseCheckersEnv(num_players=2, player_colors=["yellow", "purple"], max_turns=500)
    env.reset()

    # The same model file can be used as a plain agent or a search agent
    agent_cls = AfterstateSearchAgent if search_agent else AfterstateValueAgent
    agent = agent_cls(
        state_size=len(env.get_state_for_player("yellow")),
        player_color="yellow",
        name="AfterstateSearch" if search_agent else "Afterstate",
    )
    if not agent.load_model(str(path), verbose=False):
        raise RuntimeError("Could not load afterstate model")
    agent.epsilon = 0.0
    if search_agent:
        # These values are close to the tournament search settings
        agent.search_depth = 2
        agent.endgame_depth = 3
        agent.search_width = 8
        agent.endgame_width = 10
        agent.response_width = 5
        agent.endgame_response_width = 6
        agent.default_time_budget_seconds = tournament_player.TOURNAMENT_AFTERSTATE_BASE_BUDGET
    return configure_afterstate(agent, dense_gap=dense_gap, jump_weight=jump_weight)


def configure_search_mode(controller, turn_mode=None, opponent_mode=None):
    # Set search mode on both the 2 player and multiplayer agents
    for attr_name in ("afterstate", "multiplayer_afterstate"):
        agent = getattr(controller, attr_name, None)
        if agent is None:
            continue
        if turn_mode is not None:
            agent.search_turn_mode = str(turn_mode)
        if opponent_mode is not None:
            agent.search_opponent_mode = str(opponent_mode)
    return controller


def quiet_call(fn, *args, **kwargs):
    # Hide extra model loading text so result tables stay readable
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return fn(*args, **kwargs)


# Lets the tournament fallback search run on a local env
class TournamentFallbackController:

    def _server_state_from_env(self, env, move_count):
        # Convert local state to the same format as the server state
        pins = {}
        for colour in env.player_colors:
            positions = [
                int(pin.axialindex)
                for pin in env.pins_on_board
                if str(pin.color) == str(colour)
            ]
            pins[str(colour)] = positions
        return {
            "pins": pins,
            "players": [{"colour": str(c), "name": str(c)} for c in env.player_colors],
            "move_count": move_count,
            "current_turn_colour": str(env.get_current_player()),
            "status": "PLAYING",
        }

    def choose_action(self, env, valid_actions, move_count):
        # The tournament player expects legal moves grouped by pin id
        legal_moves = {}
        for pin_id, dest in valid_actions:
            legal_moves.setdefault(str(int(pin_id)), []).append(int(dest))

        state = self._server_state_from_env(env, move_count)
        # 2 player search gets a smaller budget than multiplayer search
        time_budget = (
            tournament_player.TWO_PLAYER_FALLBACK_TIME_BUDGET
            if len(env.player_colors) == 2
            else tournament_player.MULTI_PLAYER_FALLBACK_TIME_BUDGET
        )
        depth = (
            tournament_player.TWO_PLAYER_FALLBACK_DEPTH
            if len(env.player_colors) == 2
            else tournament_player.MULTI_PLAYER_FALLBACK_DEPTH
        )
        return quiet_call(
            tournament_player.pick_best_action,
            state,
            str(env.get_current_player()),
            legal_moves,
            depth,
            time_budget,
        )


# Lets fallback search be used as an opponent
class FallbackAgent:

    def __init__(self, name="Fallback"):
        self.name = name
        self.controller = TournamentFallbackController()

    def choose_action(self, env, valid_actions):
        return self.controller.choose_action(
            env,
            valid_actions,
            getattr(env, "turn_count", 0),
        )


# Lets tournament route controllers be used as opponents
class ControllerAgent:

    def __init__(self, controller, name="Controller"):
        self.name = name
        self.controller = controller

    def choose_action(self, env, valid_actions):
        return self.controller.choose_action(
            env,
            valid_actions,
            getattr(env, "turn_count", 0),
        )


def _mean_cartesian(board, cell_ids):
    # Lane helpers use board coordinates so they work for every colour
    points = [board.cartesian[int(cell_id)] for cell_id in cell_ids]
    if not points:
        return (0.0, 0.0)
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _distance_to_segment(point, start, end):
    # Distance from a cell to the direct home to goal lane
    px, py = point
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    cx, cy = sx + t * dx, sy + t * dy
    return math.hypot(px - cx, py - cy)


def _corridor_distance(board, colour, cell_id):
    # Each colour has its own home and target direction
    target_colour = board.colour_opposites.get(str(colour))
    if not target_colour:
        return 0.0
    home = board.axial_of_colour(str(colour))
    target = board.axial_of_colour(target_colour)
    if not home or not target:
        return 0.0
    point = board.cartesian[int(cell_id)]
    spacing = float(getattr(board, "spacing", 34) or 34)
    return _distance_to_segment(
        point,
        _mean_cartesian(board, home),
        _mean_cartesian(board, target),
    ) / spacing


def _corridor_features(env, colour, action, corridor_width=2.4):
    # Collect only the values needed by the lane check
    pin_id, dest = action
    pin = env.game.get_pin_id_of_player(env.pins_on_board, colour, str(pin_id))
    if pin is None:
        return None

    origin = int(pin.axialindex)
    dest = int(dest)
    target_colour = env.board.colour_opposites.get(str(colour), "")
    targets = set(env.board.axial_of_colour(target_colour)) if target_colour else set()
    origin_lane = _corridor_distance(env.board, colour, origin)
    dest_lane = _corridor_distance(env.board, colour, dest)
    progress = env.evaluate_action_progress(action, colour)
    jump = env.axial_distance(origin, dest)
    enters_target = dest in targets and origin not in targets
    stays_target = dest in targets and origin in targets
    return {
        # Keep these values clear so the lane check is easy to read
        "origin": origin,
        "dest": dest,
        "progress": float(progress),
        "jump": int(jump),
        "origin_lane": float(origin_lane),
        "dest_lane": float(dest_lane),
        "lane_delta": float(dest_lane - origin_lane),
        "outside_dest": max(0.0, float(dest_lane) - float(corridor_width)),
        "enters_target": enters_target,
        "stays_target": stays_target,
    }


def _corridor_score(features):
    # This score only chooses between legal moves here It does not train the model
    if features is None:
        return float("-inf")
    score = features["progress"] * 3.0
    score -= features["outside_dest"] * 4.0
    score -= max(0.0, features["lane_delta"]) * 2.0
    if features["origin_lane"] > 2.4 and features["lane_delta"] < 0:
        score += min(8.0, -features["lane_delta"] * 4.0)
    if features["dest_lane"] <= 2.4:
        score += 2.0
    if features["enters_target"]:
        score += 12.0
    elif features["stays_target"]:
        score += 4.0
    if features["jump"] >= 3:
        score += min(8.0, features["jump"] * 1.5)
    return score


def _corridor_adjust_action(env, colour, action, valid_actions, corridor_width=2.4, override_margin=6.0):
    # Keep the learned move unless a lane move is clearly better
    if action is None or not valid_actions:
        return action

    base_features = _corridor_features(env, colour, action, corridor_width=corridor_width)
    base_score = _corridor_score(base_features)
    best_action = action
    best_score = base_score

    for candidate in valid_actions:
        # Check every legal move, but replace only when clearly better
        features = _corridor_features(env, colour, candidate, corridor_width=corridor_width)
        score = _corridor_score(features)
        if score > best_score:
            best_score = score
            best_action = candidate

    margin = float(override_margin)
    if base_features is not None and (
        base_features["progress"] >= 12.0
        or base_features["enters_target"]
        or (base_features["jump"] >= 4 and base_features["progress"] >= 3.0)
    ):
        # Protect strong forward moves from being replaced
        margin += 8.0

    return best_action if best_score > base_score + margin else action


def _positions_for_colour(env, colour):
    # This uses a small local version of the tournament score
    return [
        int(pin.axialindex)
        for pin in env.pins_on_board
        if str(pin.color) == str(colour)
    ]


def _target_cells_for_colour(env, colour):
    target_colour = env.board.colour_opposites.get(str(colour), "")
    return set(env.board.axial_of_colour(target_colour)) if target_colour else set()


def _nearest_target_distance(env, cell, target_cells):
    if not target_cells:
        return 0
    return min(env.axial_distance(int(cell), int(target)) for target in target_cells)


def _score_race_pressure(env, colour, my_pins, my_distance):
    # In late game, focus more on finishing pins and catching the leader
    move_count = int(getattr(env, "turn_count", 0) or 0)
    player_count = max(1, len(getattr(env, "player_colors", ()) or ()))
    if player_count <= 2:
        midgame_turn, late_turn = 50, 80
    elif player_count == 3:
        midgame_turn, late_turn = 60, 95
    else:
        midgame_turn, late_turn = 72, 110

    pressure = 0.0
    # Time pressure grows near the usual game end
    if move_count >= midgame_turn:
        pressure = max(pressure, 0.45)
    if move_count >= late_turn:
        pressure = max(pressure, 0.85)
    if move_count >= late_turn + max(24, 8 * player_count):
        pressure = max(pressure, 1.15)

    my_score_proxy = 100.0 * float(my_pins) - 2.0 * float(my_distance)
    for opp_colour in getattr(env, "player_colors", ()) or ():
        opp_colour = str(opp_colour)
        if opp_colour == str(colour):
            continue
        opp_positions = _positions_for_colour(env, opp_colour)
        opp_targets = _target_cells_for_colour(env, opp_colour)
        if not opp_positions or not opp_targets:
            continue
        opp_pins = sum(1 for position in opp_positions if position in opp_targets)
        opp_distance = tournament_player._total_dist(opp_positions, opp_targets)
        opp_score_proxy = 100.0 * float(opp_pins) - 2.0 * float(opp_distance)
        lead = opp_score_proxy - my_score_proxy
        # If an opponent is ahead, prefer moves that finish pins or close distance
        if lead > 60.0:
            pressure = max(pressure, min(1.45, lead / 180.0))
        if opp_pins > my_pins:
            pressure = max(pressure, min(1.65, 0.75 + 0.25 * (opp_pins - my_pins)))
        elif opp_pins == my_pins and opp_distance + 8 < my_distance:
            pressure = max(pressure, 0.65)
        if opp_pins >= 8 and opp_pins >= my_pins:
            pressure = max(pressure, 1.15)

    return min(1.8, pressure)


def _score_delta_features(env, colour, action):
    # Estimate how a move changes pins in goal and distance to goal
    if action is None:
        return None
    try:
        pin_id, dest = int(action[0]), int(action[1])
    except (TypeError, ValueError, IndexError):
        return None

    pin = env.game.get_pin_id_of_player(env.pins_on_board, colour, str(pin_id))
    if pin is None:
        return None

    positions = _positions_for_colour(env, colour)
    targets = _target_cells_for_colour(env, colour)
    if not positions or not targets:
        return None

    origin = int(pin.axialindex)
    try:
        replace_index = positions.index(origin)
    except ValueError:
        return None

    before_distance = tournament_player._total_dist(positions, targets)
    before_pins = sum(1 for position in positions if position in targets)
    race_pressure = _score_race_pressure(env, colour, before_pins, before_distance)

    # Simulate only our move Opponent replies would be too slow here
    simulated = list(positions)
    simulated[replace_index] = dest
    after_distance = tournament_player._total_dist(simulated, targets)
    after_pins = sum(1 for position in simulated if position in targets)

    pin_gain = after_pins - before_pins
    distance_gain = before_distance - after_distance
    origin_target_distance = _nearest_target_distance(env, origin, targets)
    dest_target_distance = _nearest_target_distance(env, dest, targets)
    pin_distance_gain = origin_target_distance - dest_target_distance
    enters_target = dest in targets and origin not in targets
    leaves_target = origin in targets and dest not in targets
    stays_target = origin in targets and dest in targets

    score = 100.0 * pin_gain + 2.0 * distance_gain
    # Target pins are worth a lot, so entering or leaving target matters more
    if enters_target:
        score += 35.0
    if leaves_target:
        score -= 240.0
    if stays_target:
        score += 4.0
    if pin_gain <= 0 and distance_gain <= 0 and not enters_target:
        score -= 10.0

    endgame_pressure = before_pins >= max(6, len(positions) - 4)
    if endgame_pressure:
        # Late game should focus on the last pin that can score soon
        weight = 18.0 if before_pins >= len(positions) - 2 else 10.0
        if origin not in targets:
            score += weight * pin_distance_gain
            if pin_distance_gain <= 0 and not enters_target:
                score -= 22.0
        elif stays_target:
            score -= 6.0
    last_pin_pressure = before_pins >= len(positions) - 1
    if last_pin_pressure:
        # With one pin left outside, even small distance changes matter
        if origin not in targets:
            score += 42.0 * pin_distance_gain
            if before_distance <= 3 and pin_distance_gain > 0:
                score += 80.0
            if pin_distance_gain <= 0 and not enters_target:
                score -= 80.0
        elif not stays_target:
            score -= 120.0
    if race_pressure > 0.0:
        # If another player may finish first, clean up more aggressively
        if origin not in targets:
            score += race_pressure * 6.0 * pin_distance_gain
            if pin_distance_gain >= 8:
                score += race_pressure * min(24.0, 1.2 * pin_distance_gain)
            if pin_distance_gain <= 0 and not enters_target:
                score -= race_pressure * 14.0
        elif after_pins < len(positions) and not enters_target:
            score -= race_pressure * 7.0
    if after_pins == len(positions):
        score += 600.0
    elif after_pins >= len(positions) - 1 and pin_gain > 0:
        score += 120.0

    try:
        jump = int(env.axial_distance(origin, dest))
    except Exception:
        jump = 0
    if jump >= 3 and distance_gain > 0:
        score += min(10.0, 1.5 * jump)

    return {
        "score": float(score),
        "pin_gain": int(pin_gain),
        "distance_gain": float(distance_gain),
        "pin_distance_gain": float(pin_distance_gain),
        "race_pressure": float(race_pressure),
        "enters_target": bool(enters_target),
        "leaves_target": bool(leaves_target),
        "finishes_game": after_pins == len(positions),
        "pins_before": int(before_pins),
        "pins_after": int(after_pins),
        "pin_count": int(len(positions)),
        "distance_before": float(before_distance),
        "origin_target_distance": float(origin_target_distance),
        "dest_target_distance": float(dest_target_distance),
    }


def _score_safeguard_action(env, colour, action, valid_actions):
    # Final safety check stay in target, finish pins, and clean up late game
    if action is None or not valid_actions:
        return action

    base_features = _score_delta_features(env, colour, action)
    if base_features is None:
        return action

    best_action = action
    best_features = base_features
    cleanup_action = None
    cleanup_features = None
    for candidate in valid_actions:
        features = _score_delta_features(env, colour, candidate)
        if features is not None and features["score"] > best_features["score"]:
            best_action = candidate
            best_features = features
        if (
            features is not None
            and base_features["pins_before"] >= max(8, base_features["pin_count"] - 2)
            and features["pin_distance_gain"] >= base_features["pin_distance_gain"] + 2.0
            and features["score"] >= base_features["score"] - 6.0
            and not features["leaves_target"]
        ):
            # Setup moves can help the last outside pin score next turn
            cleanup_rank = (
                features["pins_after"],
                features["origin_target_distance"],
                features["pin_distance_gain"],
                features["score"],
            )
            previous_rank = (
                cleanup_features["pins_after"],
                cleanup_features["origin_target_distance"],
                cleanup_features["pin_distance_gain"],
                cleanup_features["score"],
            ) if cleanup_features is not None else None
            if previous_rank is None or cleanup_rank > previous_rank:
                cleanup_action = candidate
                cleanup_features = features

    margin = (
        tournament_player.SCORE_SAFEGUARD_BAD_MARGIN
        if base_features["score"] < 0
        else tournament_player.SCORE_SAFEGUARD_OVERRIDE_MARGIN
    )
    # Some improvements are strong enough to lower the replace limit
    if base_features["finishes_game"]:
        return action
    if best_features["finishes_game"]:
        margin = -1.0
    if best_features["pin_gain"] > base_features["pin_gain"]:
        margin = min(margin, 4.0)
    if best_features["enters_target"] and not base_features["enters_target"]:
        margin = min(margin, 6.0)
    if (
        base_features["pins_before"] >= 6
        and best_features["pin_distance_gain"] > base_features["pin_distance_gain"]
    ):
        margin = min(margin, 2.0)
    if (
        base_features["pins_before"] >= max(8, base_features["pin_count"] - 2)
        and best_features["pin_distance_gain"] > base_features["pin_distance_gain"]
    ):
        margin = min(margin, 1.0)
    race_pressure = max(base_features.get("race_pressure", 0.0), best_features.get("race_pressure", 0.0))
    if (
        race_pressure >= 0.75
        and best_features["pin_distance_gain"] > base_features["pin_distance_gain"]
    ):
        margin = min(margin, 3.0 if race_pressure < 1.15 else 1.5)
    if (
        base_features["pins_before"] >= base_features["pin_count"] - 1
        and best_features["pin_distance_gain"] > base_features["pin_distance_gain"]
    ):
        margin = min(margin, 0.5)
        if base_features["distance_before"] <= 3:
            margin = -0.5

    if cleanup_action is not None:
        # Prefer a strong cleanup move over the normal best move
        return cleanup_action

    return best_action if best_features["score"] > base_features["score"] + margin else action


# Local version of the final tournament route
class TournamentRoutingController:

    def __init__(self, dense_gap=None, jump_weight=None, model_path=None, search_agent=False):
        # By default, use the 2p model for 2p and multiplayer model otherwise
        self.model_override = model_path is not None
        self.search_agent = bool(search_agent)
        self.afterstate = load_afterstate(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            search_agent=search_agent,
        )
        self.multiplayer_afterstate = self.afterstate
        if model_path is None and Path(AFTERSTATE_MULTIPLAYER_MODEL).exists():
            self.multiplayer_afterstate = load_afterstate(
                dense_gap=dense_gap,
                jump_weight=jump_weight,
                model_path=AFTERSTATE_MULTIPLAYER_MODEL,
                search_agent=search_agent,
            )
        self.fallback = TournamentFallbackController()

    def _select_afterstate(self, env):
        # Keep both models loaded and choose by player count
        if len(env.player_colors) > 2 and self.multiplayer_afterstate is not None:
            return self.multiplayer_afterstate
        return self.afterstate

    def choose_action(self, env, valid_actions, move_count):
        # Use the same move choice route as the tournament player
        state = self.fallback._server_state_from_env(env, move_count)
        current = str(env.get_current_player())
        sprint_pressure = tournament_player._sprint_pressure(state, current)
        if sprint_pressure >= tournament_player.TOURNAMENT_SPRINT_TRIGGER:
            # Sprint mode is a direct fix for urgent endgames
            action = tournament_player._sprint_action(state, current, valid_actions, sprint_pressure)
            return _score_safeguard_action(env, current, action, valid_actions)

        agent = self._select_afterstate(env)
        if agent is not None and tournament_player.should_use_learned_agent(state, current):
            # Normal route ask the learned agent with the same time budget
            action = agent.choose_action(
                env,
                valid_actions,
                time_budget_seconds=tournament_player._afterstate_time_budget(state, current),
            )
        else:
            # Fallback is still available for unsafe positions
            action = self.fallback.choose_action(env, valid_actions, move_count)
        return _score_safeguard_action(env, current, action, valid_actions)


# Starts with duel search, then switches to full cycle search
class SearchSettleEarlyRoutingController:

    def __init__(
        self,
        dense_gap=None,
        jump_weight=None,
        model_path=None,
        switch_turn=60,
        turn_limit=None,
        min_players=4,
    ):
        self.default = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            search_agent=False,
        )
        self.duel_search = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            search_agent=True,
        )
        self.cycle_search = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            search_agent=True,
        )
        configure_search_mode(self.duel_search, "duel", "first")
        configure_search_mode(self.cycle_search, "cycle", "leader")
        self.fallback = TournamentFallbackController()
        self.switch_turn = int(switch_turn)
        self.turn_limit = turn_limit
        self.min_players = min_players

    def choose_action(self, env, valid_actions, move_count=None):
        if move_count is None:
            move_count = getattr(env, "turn_count", 0)
        state = self.fallback._server_state_from_env(env, move_count)
        # Early game uses duel search Later game uses full cycle search
        if not tournament_player._search_hybrid_early_trigger(
            state,
            turn_limit=self.turn_limit,
            min_players=self.min_players,
        ):
            return self.default.choose_action(env, valid_actions, move_count)
        if int(state.get("move_count", move_count) or 0) >= self.switch_turn:
            return self.cycle_search.choose_action(env, valid_actions, move_count)
        return self.duel_search.choose_action(env, valid_actions, move_count)


# Uses 2p search, settle search in multiplayer, and safer 6p seats
class SearchSeatBaseRoutingController:

    def __init__(
        self,
        dense_gap=None,
        jump_weight=None,
        model_path=None,
        base_model_path=None,
        switch_turn=75,
        base_seats=(3, 5),
        base_colours=("blue", "purple"),
    ):
        self.settle75 = SearchSettleEarlyRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            switch_turn=switch_turn,
            min_players=4,
        )
        self.two_player_search = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            search_agent=True,
        )
        if base_model_path is None:
            base_model_path = AFTERSTATE_TWOPLAYER_MODEL if model_path is None else model_path
        self.base = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=base_model_path,
            search_agent=False,
        )
        self.base_seats = {int(seat) for seat in base_seats}
        self.base_colours = {str(colour) for colour in base_colours}

    def _use_base_route(self, env):
        # Some 6p seats were weaker, so they use the calmer base route
        if len(getattr(env, "player_colors", ()) or ()) != 6:
            return False
        current = str(env.get_current_player())
        colours = [str(colour) for colour in env.player_colors]
        try:
            seat = colours.index(current)
        except ValueError:
            seat = None
        return seat in self.base_seats or current in self.base_colours

    def choose_action(self, env, valid_actions, move_count=None):
        if move_count is None:
            move_count = getattr(env, "turn_count", 0)
        if len(getattr(env, "player_colors", ()) or ()) == 2:
            # In 2p, pure search is strongest
            return self.two_player_search.choose_action(env, valid_actions, move_count)
        if self._use_base_route(env):
            return self.base.choose_action(env, valid_actions, move_count)
        return self.settle75.choose_action(env, valid_actions, move_count)


# Seat safe route with leader pressure late in the game
# Early game uses the safer settle route
# Later it chases the leader only when needed
class SearchSeatMixRoutingController:

    def __init__(
        self,
        dense_gap=None,
        jump_weight=None,
        model_path=None,
        base_model_path=None,
        switch_turn=75,
        turn_limit=None,
        min_players=4,
        leader_pin_gap=1,
        leader_dist_gap=10,
        base_seats=(3, 5),
        base_colours=("blue", "purple"),
    ):
        self.default = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            search_agent=False,
        )
        self.duel_search = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            search_agent=True,
        )
        self.cycle_search = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            search_agent=True,
        )
        configure_search_mode(self.duel_search, "duel", "first")
        configure_search_mode(self.cycle_search, "cycle", "leader")
        self.best_reply = SearchBestReplyRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            min_players=min_players,
        )
        if base_model_path is None:
            base_model_path = AFTERSTATE_TWOPLAYER_MODEL if model_path is None else model_path
        self.base = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=base_model_path,
            search_agent=False,
        )
        self.fallback = TournamentFallbackController()
        self.switch_turn = int(switch_turn)
        self.turn_limit = turn_limit
        self.min_players = int(min_players)
        self.leader_pin_gap = int(leader_pin_gap)
        self.leader_dist_gap = int(leader_dist_gap)
        self.base_seats = {int(seat) for seat in base_seats}
        self.base_colours = {str(colour) for colour in base_colours}

    def _use_base_route(self, env):
        if len(getattr(env, "player_colors", ()) or ()) != 6:
            return False
        current = str(env.get_current_player())
        colours = [str(colour) for colour in env.player_colors]
        try:
            seat = colours.index(current)
        except ValueError:
            seat = None
        return seat in self.base_seats or current in self.base_colours

    def _should_pressure_leader(self, env):
        # Chase the leader only when another colour is clearly ahead
        colours = [str(colour) for colour in getattr(env, "player_colors", ()) or ()]
        if len(colours) < self.min_players:
            return False
        current = str(env.get_current_player())
        opponents = [colour for colour in colours if colour != current]
        if not opponents:
            return False
        leader = max(
            # Leader means most pins, then lowest distance
            opponents,
            key=lambda colour: (
                env.count_player_pins_in_target(colour),
                -env.total_distance_to_target(colour),
            ),
        )
        my_pins = env.count_player_pins_in_target(current)
        leader_pins = env.count_player_pins_in_target(leader)
        my_dist = env.total_distance_to_target(current)
        leader_dist = env.total_distance_to_target(leader)
        return (
            leader_pins >= my_pins + self.leader_pin_gap
            or leader_dist <= my_dist - self.leader_dist_gap
        )

    def choose_action(self, env, valid_actions, move_count=None):
        if move_count is None:
            move_count = getattr(env, "turn_count", 0)
        if len(getattr(env, "player_colors", ()) or ()) == 2:
            # Two player games are a duel, so use duel search
            return self.duel_search.choose_action(env, valid_actions, move_count)
        if self._use_base_route(env):
            return self.base.choose_action(env, valid_actions, move_count)

        state = self.fallback._server_state_from_env(env, move_count)
        if not tournament_player._search_hybrid_early_trigger(
            state,
            turn_limit=self.turn_limit,
            min_players=self.min_players,
        ):
            return self.default.choose_action(env, valid_actions, move_count)

        current_move = int(state.get("move_count", move_count) or 0)
        if current_move < self.switch_turn:
            # Opening focus the first opponent
            return self.duel_search.choose_action(env, valid_actions, move_count)
        if self._should_pressure_leader(env):
            # Mid and late game chase the leader only when needed
            return self.best_reply.choose_action(env, valid_actions, move_count)
        return self.cycle_search.choose_action(env, valid_actions, move_count)


# Soft lane based route
# The main route picks first
# This only replaces moves when a legal lane move is clearly better
class SearchCorridorRoutingController:

    def __init__(
        self,
        primary,
        corridor_width=2.4,
        override_margin=6.0,
        seat_safe=False,
    ):
        self.primary = primary
        self.corridor_width = float(corridor_width)
        self.override_margin = float(override_margin)
        self.seat_safe = bool(seat_safe)

    def choose_action(self, env, valid_actions, move_count=None):
        if move_count is None:
            move_count = getattr(env, "turn_count", 0)
        action = self.primary.choose_action(env, valid_actions, move_count)
        if self.seat_safe and hasattr(self.primary, "_use_base_route") and self.primary._use_base_route(env):
            # Do not replace the safer route on weak 6p seats
            return action
        current = str(env.get_current_player())
        return _corridor_adjust_action(
            env,
            current,
            action,
            valid_actions,
            corridor_width=self.corridor_width,
            override_margin=self.override_margin,
        )


# Uses lane bias in 4p and the safer seatbase route otherwise
class SearchFourCorridorSeatBaseRoutingController:

    def __init__(
        self,
        dense_gap=None,
        jump_weight=None,
        model_path=None,
        base_model_path=None,
        switch_turn=75,
    ):
        self.corridor = SearchCorridorRoutingController(
            SearchSettleEarlyRoutingController(
                dense_gap=dense_gap,
                jump_weight=jump_weight,
                model_path=model_path,
                switch_turn=switch_turn,
                min_players=4,
            )
        )
        self.seatbase = SearchSeatBaseRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            base_model_path=base_model_path,
            switch_turn=switch_turn,
        )

    def choose_action(self, env, valid_actions, move_count=None):
        player_count = len(getattr(env, "player_colors", ()) or ())
        if player_count == 4:
            # Lane bias helped mostly in 4p 6p keeps the safer route
            return self.corridor.choose_action(env, valid_actions, move_count)
        return self.seatbase.choose_action(env, valid_actions, move_count)


# Best reply style route for 4p and 6p
# It uses the current leader as the duel target
class SearchBestReplyRoutingController:

    def __init__(
        self,
        dense_gap=None,
        jump_weight=None,
        model_path=None,
        min_players=4,
    ):
        self.default = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            search_agent=False,
        )
        self.best_reply = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            search_agent=True,
        )
        configure_search_mode(self.best_reply, "duel", "first")
        self.min_players = int(min_players)

    def _active_search_agent(self, env):
        # Use the same model split as the base controller
        if len(getattr(env, "player_colors", ()) or ()) > 2:
            return self.best_reply.multiplayer_afterstate
        return self.best_reply.afterstate

    def _best_reply_player_order(self, env):
        # Search starts with us, then the main threat, then everyone else
        colours = [str(colour) for colour in getattr(env, "player_colors", ()) or ()]
        current = str(env.get_current_player())
        if current not in colours:
            return colours

        opponents = [colour for colour in colours if colour != current]
        if not opponents:
            return [current]

        leader = max(
            # Same leader rule as elsewhere pins, then distance
            opponents,
            key=lambda colour: (
                env.count_player_pins_in_target(colour),
                -env.total_distance_to_target(colour),
            ),
        )
        return [current, leader] + [
            colour for colour in colours
            if colour not in {current, leader}
        ]

    def choose_action(self, env, valid_actions, move_count=None):
        if move_count is None:
            move_count = getattr(env, "turn_count", 0)
        if len(getattr(env, "player_colors", ()) or ()) < self.min_players:
            return self.default.choose_action(env, valid_actions, move_count)

        state = self.default.fallback._server_state_from_env(env, move_count)
        current = str(env.get_current_player())
        agent = self._active_search_agent(env)
        if agent is None or not tournament_player.should_use_learned_agent(state, current):
            # If the learned agent is unsafe here, keep the normal route
            return self.default.choose_action(env, valid_actions, move_count)

        # Pass the chosen player order into search
        return agent.choose_action_from_board(
            env.pins_on_board,
            current,
            valid_actions,
            env.board,
            time_budget_seconds=tournament_player._afterstate_time_budget(state, current),
            turn_count=move_count,
            move_history=getattr(env, "move_history", None),
            player_order=self._best_reply_player_order(env),
        )


# Seat safe best reply route
# Weak 6p seats use the safer base route
class SearchSeatBestReplyRoutingController:

    def __init__(
        self,
        dense_gap=None,
        jump_weight=None,
        model_path=None,
        base_seats=(3, 5),
        base_colours=("blue", "purple"),
    ):
        self.best_reply = SearchBestReplyRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            min_players=4,
        )
        base_model_path = AFTERSTATE_TWOPLAYER_MODEL if model_path is None else model_path
        self.base = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=base_model_path,
            search_agent=False,
        )
        self.base_seats = {int(seat) for seat in base_seats}
        self.base_colours = {str(colour) for colour in base_colours}

    def _use_base_route(self, env):
        # Same weak seat rule as seatbase, with best reply search elsewhere
        if len(getattr(env, "player_colors", ()) or ()) != 6:
            return False
        current = str(env.get_current_player())
        colours = [str(colour) for colour in env.player_colors]
        try:
            seat = colours.index(current)
        except ValueError:
            seat = None
        return seat in self.base_seats or current in self.base_colours

    def choose_action(self, env, valid_actions, move_count=None):
        if move_count is None:
            move_count = getattr(env, "turn_count", 0)
        if self._use_base_route(env):
            return self.base.choose_action(env, valid_actions, move_count)
        return self.best_reply.choose_action(env, valid_actions, move_count)

def build_frozen_opponent(kind):
    # Learned opponents are loaded once and reused for every game
    kind = str(kind).strip().lower()
    if kind in {"afterstate", "challenger"}:
        # afterstate and challenger both mean the final multiplayer model
        agent = load_afterstate(model_path=AFTERSTATE_MULTIPLAYER_MODEL)
        agent.epsilon = 0.0
        return agent
    if kind == "original":
        # Original is kept to test the old 2 player only model
        agent = load_afterstate(model_path=AFTERSTATE_TWOPLAYER_MODEL)
        agent.epsilon = 0.0
        return agent
    if kind == "external_afterstate":
        agent = load_afterstate(model_path=AFTERSTATE_EXTERNAL_MODEL)
        agent.epsilon = 0.0
        return agent
    if kind == "fallback":
        return FallbackAgent(name="Fallback")
    if kind == "tournament":
        # A fixed tournament route can be used as an opponent
        return ControllerAgent(TournamentRoutingController(), name="TournamentRoute")
    if kind == "settle75":
        return ControllerAgent(
            SearchSettleEarlyRoutingController(switch_turn=75, min_players=4),
            name="Search46Settle75",
        )
    if kind == "seatbase":
        return ControllerAgent(
            SearchSeatBaseRoutingController(switch_turn=75),
            name="Search46SeatBase",
        )
    if kind == "seatmix":
        return ControllerAgent(
            SearchSeatMixRoutingController(switch_turn=75),
            name="Search46SeatMix",
        )
    if kind == "corridor":
        return ControllerAgent(
            SearchCorridorRoutingController(
                SearchSettleEarlyRoutingController(switch_turn=75, min_players=4),
            ),
            name="Search46Corridor",
        )
    if kind == "seatcorridor":
        return ControllerAgent(
            SearchCorridorRoutingController(
                SearchSeatBaseRoutingController(switch_turn=75),
                seat_safe=True,
            ),
            name="Search46SeatCorridor",
        )
    if kind == "corridor4seatbase":
        return ControllerAgent(
            SearchFourCorridorSeatBaseRoutingController(switch_turn=75),
            name="Search4Corridor6SeatBase",
        )
    if kind == "brs":
        return ControllerAgent(
            SearchBestReplyRoutingController(),
            name="Search46BRS",
        )
    if kind == "seatbrs":
        return ControllerAgent(
            SearchSeatBestReplyRoutingController(),
            name="Search46SeatBRS",
        )
    raise ValueError(
        "unsupported frozen opponent "
        f"{kind!r}; choose from {', '.join(SUPPORTED_OPPONENT_KINDS)}"
    )


def run_one_match(controller, num_players, opponent_factory, max_turns=300, agent_seat=0):
    # The tested controller gets one seat Opponents fill the rest
    env = ChineseCheckersEnv(num_players=num_players, max_turns=max_turns)
    quiet_call(env.reset)
    player_colors = list(env.player_colors)
    my_colour = player_colors[agent_seat]
    opponents = {c: opponent_factory(c) for i, c in enumerate(player_colors) if i != agent_seat}

    done = False
    info = {}
    moves = 0
    my_moves = 0

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            info = {"message": "No valid actions"}
            break

        current = env.get_current_player()
        if current == my_colour:
            # Search routes need move count because some switch after opening
            if isinstance(controller, (
                TournamentFallbackController,
                TournamentRoutingController,
                SearchSettleEarlyRoutingController,
                SearchSeatBaseRoutingController,
                SearchSeatMixRoutingController,
                SearchCorridorRoutingController,
                SearchFourCorridorSeatBaseRoutingController,
                SearchBestReplyRoutingController,
                SearchSeatBestReplyRoutingController,
            )):
                action = controller.choose_action(env, valid_actions, moves)
            else:
                action = controller.choose_action(env, valid_actions)
            my_moves += 1
        else:
            # Opponents use the normal agent API
            action = opponents[current].choose_action(env, valid_actions)

        if action is None:
            info = {"message": "No action selected"}
            break

        _, _, done, info = env.step(action)
        moves += 1

    # Score every player and rank the tested controller
    scoreboard = []
    for colour in player_colors:
        pins = env.count_player_pins_in_target(colour)
        dist = env.total_distance_to_target(colour)
        # Local scoring matches the tournament formula without time score
        won = (info.get("message") == f"{colour} wins")
        pin_score = pins * 100.0
        dist_score = max(0.0, 400.0 - 2 * dist) if moves > 0 else 0.0
        win_bonus = 1000.0 if won else 0.0
        total = pin_score + dist_score + win_bonus
        scoreboard.append({
            "colour": colour,
            "pins": pins,
            "distance": dist,
            "won": won,
            "score": total,
        })

    scoreboard.sort(key=lambda s: s["score"], reverse=True)
    my_rank = next(i for i, s in enumerate(scoreboard) if s["colour"] == my_colour) + 1
    my_record = next(s for s in scoreboard if s["colour"] == my_colour)

    return {
        "num_players":  num_players,
        "my_colour":    my_colour,
        "my_rank":      my_rank,
        "my_pins":      my_record["pins"],
        "my_distance":  my_record["distance"],
        "my_score":     my_record["score"],
        "my_won":       my_record["won"],
        "total_moves":  moves,
        "my_moves":     my_moves,
        "agent_seat":    agent_seat,
        "result":       info.get("message", "Unknown"),
    }


def evaluate(label, controller, num_players, opponent_factory, games=10, max_turns=300):
    print(f"\n{label}")
    # Cycle seats so one colour does not decide the result
    results = [
        run_one_match(controller, num_players, opponent_factory, max_turns=max_turns, agent_seat=i % num_players)
        for i in range(games)
    ]

    wins = sum(1 for r in results if r["my_won"])
    avg_rank = sum(r["my_rank"] for r in results) / games
    avg_pins = sum(r["my_pins"] for r in results) / games
    avg_score = sum(r["my_score"] for r in results) / games
    avg_dist = sum(r["my_distance"] for r in results) / games
    rank_dist = {r: sum(1 for x in results if x["my_rank"] == r) for r in range(1, num_players + 1)}

    print(f"  games={games}  wins={wins}  win_rate={wins/games*100:.0f}%")
    print(f"  avg rank:    {avg_rank:.2f} (1 = best, {num_players} = worst)")
    print(f"  avg pins:    {avg_pins:.2f} / 10")
    print(f"  avg distance: {avg_dist:.2f}")
    print(f"  avg score:   {avg_score:.1f}")
    print("  rank histogram:", " ".join(f"{r}:{c}" for r, c in rank_dist.items()))

    if RUN_DETAIL:
        # Details help find colour or seat weakness
        print("  by colour:")
        for colour in sorted({r["my_colour"] for r in results}):
            subset = [r for r in results if r["my_colour"] == colour]
            colour_wins = sum(1 for r in subset if r["my_won"])
            colour_rank = sum(r["my_rank"] for r in subset) / len(subset)
            colour_pins = sum(r["my_pins"] for r in subset) / len(subset)
            print(
                f"    {colour}: games={len(subset)} wins={colour_wins} "
                f"avg_rank={colour_rank:.2f} avg_pins={colour_pins:.2f}"
            )
        print("  by seat:")
        for seat in range(num_players):
            subset = [r for r in results if r["agent_seat"] == seat]
            if not subset:
                continue
            seat_wins = sum(1 for r in subset if r["my_won"])
            seat_rank = sum(r["my_rank"] for r in subset) / len(subset)
            seat_pins = sum(r["my_pins"] for r in subset) / len(subset)
            print(
                f"    seat {seat}: games={len(subset)} wins={seat_wins} "
                f"avg_rank={seat_rank:.2f} avg_pins={seat_pins:.2f}"
            )


def main():
    # Short command form
    # benchmark script players opponent games controller
    argv = sys.argv[1:]
    positional_args = _positional_args(argv)
    num_players = int(positional_args[0]) if len(positional_args) > 0 else 4
    opponent_kind = positional_args[1].lower() if len(positional_args) > 1 else "random"
    games = int(positional_args[2]) if len(positional_args) > 2 else 12
    controller_kind = positional_args[3].lower() if len(positional_args) > 3 else "afterstate"
    seed_arg = _flag_value(argv, "--seed")
    seed = int(seed_arg) if seed_arg is not None else None
    dense_gap = _flag_value(argv, "--dense-gap")
    jump_weight = _flag_value(argv, "--jump-weight")
    model_path = _flag_value(argv, "--model")
    seed_everything(seed)

    if num_players not in (2, 3, 4, 5, 6):
        print(f"Unsupported player count {num_players}; pick 2, 3, 4, 5, or 6.")
        return
    if controller_kind == "fallback":
        controller = TournamentFallbackController()
    elif controller_kind in ("search46settle", "settle", "duelcycle") or (
        controller_kind.startswith("search46settle")
        and controller_kind.removeprefix("search46settle").isdigit()
    ):
        suffix = controller_kind.removeprefix("search46settle")
        controller = SearchSettleEarlyRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            switch_turn=int(suffix) if suffix.isdigit() else 60,
            min_players=4,
        )
    elif controller_kind in ("search46seatbase", "seatbase", "latebase") or (
        controller_kind.startswith("search46seatbase")
        and controller_kind.removeprefix("search46seatbase").isdigit()
    ):
        suffix = controller_kind.removeprefix("search46seatbase")
        controller = SearchSeatBaseRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            switch_turn=int(suffix) if suffix.isdigit() else 75,
        )
    elif controller_kind in ("search46seatmix", "seatmix") or (
        controller_kind.startswith("search46seatmix")
        and controller_kind.removeprefix("search46seatmix").isdigit()
    ):
        suffix = controller_kind.removeprefix("search46seatmix")
        switch_turn = int(suffix) if suffix.isdigit() else 75
        controller = SearchSeatMixRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            switch_turn=switch_turn,
            turn_limit=max(90, switch_turn),
        )
    elif controller_kind in ("search46corridor", "corridor") or (
        controller_kind.startswith("search46corridor")
        and controller_kind.removeprefix("search46corridor").isdigit()
    ):
        suffix = controller_kind.removeprefix("search46corridor")
        controller = SearchCorridorRoutingController(
            SearchSettleEarlyRoutingController(
                dense_gap=dense_gap,
                jump_weight=jump_weight,
                model_path=model_path,
                switch_turn=int(suffix) if suffix.isdigit() else 75,
                min_players=4,
            )
        )
    elif controller_kind in ("search46seatcorridor", "seatcorridor") or (
        controller_kind.startswith("search46seatcorridor")
        and controller_kind.removeprefix("search46seatcorridor").isdigit()
    ):
        suffix = controller_kind.removeprefix("search46seatcorridor")
        controller = SearchCorridorRoutingController(
            SearchSeatBaseRoutingController(
                dense_gap=dense_gap,
                jump_weight=jump_weight,
                model_path=model_path,
                switch_turn=int(suffix) if suffix.isdigit() else 75,
            ),
            seat_safe=True,
        )
    elif controller_kind in ("search4corridor6seatbase", "corridor4seatbase", "lane4seat6") or (
        controller_kind.startswith("search4corridor6seatbase")
        and controller_kind.removeprefix("search4corridor6seatbase").isdigit()
    ):
        suffix = controller_kind.removeprefix("search4corridor6seatbase")
        controller = SearchFourCorridorSeatBaseRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            switch_turn=int(suffix) if suffix.isdigit() else 75,
        )
    elif controller_kind in ("search46brs", "brs"):
        controller = SearchBestReplyRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            min_players=4,
        )
    elif controller_kind in ("search46seatbrs", "seatbrs"):
        controller = SearchSeatBestReplyRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
        )
    elif controller_kind == "tournament":
        controller = SearchSeatBaseRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            switch_turn=75,
        )
    elif controller_kind == "search":
        controller = TournamentRoutingController(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            search_agent=True,
        )
    elif controller_kind == "afterstate":
        controller = load_afterstate(
            dense_gap=dense_gap,
            jump_weight=jump_weight,
            model_path=model_path,
            search_agent=False,
        )
    else:
        print(
            "Unsupported controller "
            f"{controller_kind!r}; use afterstate, tournament, search, "
            "fallback, search46settle*, search46seatbase*, search46seatmix*, "
            "search46corridor*, search46seatcorridor*, "
            "search4corridor6seatbase*, "
            "search46brs, or search46seatbrs."
        )
        return

    # Make simple opponents per colour Load learned opponents once
    if opponent_kind == "greedy":
        factory = lambda colour: GreedyAgent(name=f"Greedy-{colour}")
        opponent_label = "Greedy"
    elif opponent_kind == "random":
        factory = lambda colour: RandomAgent(name=f"Random-{colour}")
        opponent_label = "Random"
    else:
        frozen = build_frozen_opponent(opponent_kind)
        factory = lambda colour: frozen
        opponent_label = getattr(frozen, "name", opponent_kind)

    # Clear labels make logs and screenshots easier to use in the report
    if controller_kind == "fallback":
        controller_label = "Handcrafted fallback"
    elif controller_kind.startswith("search46settle") or controller_kind in ("settle", "duelcycle"):
        controller_label = "Afterstate Search Settle"
    elif controller_kind.startswith("search46seatbase") or controller_kind in ("seatbase", "latebase"):
        controller_label = "Afterstate Search SeatBase"
    elif controller_kind.startswith("search46seatmix") or controller_kind == "seatmix":
        controller_label = "Afterstate Search SeatMix"
    elif controller_kind.startswith("search46seatcorridor") or controller_kind == "seatcorridor":
        controller_label = "Afterstate Search SeatCorridor"
    elif controller_kind.startswith("search46corridor") or controller_kind == "corridor":
        controller_label = "Afterstate Search Corridor"
    elif controller_kind.startswith("search4corridor6seatbase") or controller_kind in ("corridor4seatbase", "lane4seat6"):
        controller_label = "Afterstate Search Corridor/SeatBase"
    elif controller_kind in ("search46brs", "brs"):
        controller_label = "Afterstate Search BRS"
    elif controller_kind in ("search46seatbrs", "seatbrs"):
        controller_label = "Afterstate Search SeatBRS"
    elif controller_kind == "search":
        controller_label = "Afterstate search route"
    elif controller_kind == "tournament":
        controller_label = "Afterstate tournament route"
    elif controller_kind == "tournament" and model_path is not None:
        controller_label = "Tournament route model override"
    elif controller_kind == "tournament":
        controller_label = "Tournament route"
    else:
        controller_label = "Afterstate"
    label = f"{controller_label} vs {num_players - 1}×{opponent_label} ({num_players}-player, n={games})"

    try:
        evaluate(label, controller, num_players, factory, games=games, max_turns=400)
    finally:
        if RUN_NOTIFY:
            notify_finished(message=f"Multiplayer evaluation finished ({num_players}-player, {controller_label.lower()})")


if __name__ == "__main__":
    main()

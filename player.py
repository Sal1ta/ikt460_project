# Tournament client for the external game server

import os
import json
import math
import random
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.board import HexBoard
from src.paths import (
    AFTERSTATE_TWOPLAYER_MODEL,
    AFTERSTATE_FINAL_MODEL,
    AFTERSTATE_MULTIPLAYER_MODEL,
    AFTERSTATE_TRAINED_MODEL,
    first_existing,
)

# Tournament server connection settings
HOST = "10.245.30.144"
PORT = 50555

# The learned two player agents all use the same 363 feature board view
# own occupancy, opponent occupancy, and target cells
STATE_SIZE = 363
SUPPORTED_AFTERSTATE_PLAYER_COUNTS = {2, 3, 4, 5, 6}
# Fixed route environment variables must not silently change tournament play
AFTERSTATE_SEARCH_HYBRID_MODE = "search46seatmix120"

def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

TOURNAMENT_VERBOSE_DASHBOARD = _env_flag("CC_VERBOSE_DASHBOARD")
TOURNAMENT_VERBOSE_DECISIONS = _env_flag("CC_VERBOSE_DECISIONS")

def _decision_log(message):
    if TOURNAMENT_VERBOSE_DECISIONS:
        print(message)

def _parse_search_settle_switch(mode, default=60):
    for prefix, prefix_default in (
        ("search4corridor6seatbase", 75),
        ("corridor4seatbase", 75),
        ("search46seatcorridor", 75),
        ("seatcorridor", 75),
        ("search46corridor", 75),
        ("corridor", 75),
        ("search46seatbase", 75),
        ("seatbase", 75),
        ("latebase", 75),
        ("search46seatmix", 75),
        ("seatmix", 75),
        ("search46settle", default),
        ("settle", default),
        ("duelcycle", default),
    ):
        if mode == prefix:
            return int(prefix_default)
        if mode.startswith(prefix):
            suffix = mode[len(prefix):]
            if suffix.isdigit():
                return int(suffix)
    return int(default)

def _mode_matches_numbered_prefix(mode, prefixes):
    return any(
        mode == prefix
        or (mode.startswith(prefix) and mode[len(prefix):].isdigit())
        for prefix in prefixes
    )

USE_AFTERSTATE_SEARCH_SEATBASE = _mode_matches_numbered_prefix(
    AFTERSTATE_SEARCH_HYBRID_MODE,
    ("search46seatbase", "seatbase", "latebase"),
)
USE_AFTERSTATE_SEARCH_4CORRIDOR6SEATBASE = _mode_matches_numbered_prefix(
    AFTERSTATE_SEARCH_HYBRID_MODE,
    ("search4corridor6seatbase", "corridor4seatbase"),
)
USE_AFTERSTATE_SEARCH_SEATCORRIDOR = _mode_matches_numbered_prefix(
    AFTERSTATE_SEARCH_HYBRID_MODE,
    ("search46seatcorridor", "seatcorridor"),
)
USE_AFTERSTATE_SEARCH_CORRIDOR = _mode_matches_numbered_prefix(
    AFTERSTATE_SEARCH_HYBRID_MODE,
    ("search46corridor", "corridor"),
)
USE_AFTERSTATE_SEARCH_SEATMIX = _mode_matches_numbered_prefix(
    AFTERSTATE_SEARCH_HYBRID_MODE,
    ("search46seatmix", "seatmix"),
)
USE_AFTERSTATE_SEARCH_SETTLE = _mode_matches_numbered_prefix(
    AFTERSTATE_SEARCH_HYBRID_MODE,
    ("search46settle", "settle", "duelcycle"),
)
USE_AFTERSTATE_SEARCH_BRS = AFTERSTATE_SEARCH_HYBRID_MODE in {
    "search46brs",
    "brs",
}
USE_AFTERSTATE_SEARCH_SEATBRS = AFTERSTATE_SEARCH_HYBRID_MODE in {
    "search46seatbrs",
    "seatbrs",
}
AFTERSTATE_TOURNAMENT_COLORS = {"yellow", "purple", "red", "blue", "lawn green", "gray0"}
TWO_PLAYER_FALLBACK_DEPTH = 3
MULTI_PLAYER_FALLBACK_DEPTH = 3  # iterative deepening respects deadline no downside raising this
TWO_PLAYER_FALLBACK_TIME_BUDGET = 0.65
MULTI_PLAYER_FALLBACK_TIME_BUDGET = 0.08
TOURNAMENT_AFTERSTATE_DEPTH = 2
TOURNAMENT_AFTERSTATE_ENDGAME_DEPTH = 3
TOURNAMENT_AFTERSTATE_WIDTH = 8
TOURNAMENT_AFTERSTATE_ENDGAME_WIDTH = 10
TOURNAMENT_AFTERSTATE_RESPONSE_WIDTH = 5
TOURNAMENT_AFTERSTATE_ENDGAME_RESPONSE_WIDTH = 6
TOURNAMENT_AFTERSTATE_CRUISE_BUDGET = 0.10
TOURNAMENT_AFTERSTATE_BASE_BUDGET = 0.22
TOURNAMENT_AFTERSTATE_PANIC_BUDGET = 0.36
TOURNAMENT_SPRINT_TRIGGER = 0.75
TOURNAMENT_HARD_SPRINT_TRIGGER = 1.15
TOURNAMENT_SPRINT_TIME_START = 0.40
TOURNAMENT_SPRINT_TIME_FULL = 0.65
TOURNAMENT_SPRINT_TURN_START = 18.0
TOURNAMENT_SPRINT_TURN_FULL = 34.0
TOURNAMENT_POLL_SLEEP_SECONDS = 0.07
SEARCH_SETTLE_SWITCH_TURN = _parse_search_settle_switch(AFTERSTATE_SEARCH_HYBRID_MODE)
SEARCH_EARLY_TURN_LIMIT = (
    max(90, SEARCH_SETTLE_SWITCH_TURN)
    if USE_AFTERSTATE_SEARCH_SEATMIX
    else 90
)
SEARCH_SEATBASE_BASE_SEATS = {3, 5}
SEARCH_SEATBASE_BASE_COLOURS = {"blue", "purple"}
SEARCH_SEATMIX_LEADER_PIN_GAP = 1
SEARCH_SEATMIX_LEADER_DIST_GAP = 10
SEARCH_CORRIDOR_WIDTH = 2.4
SEARCH_CORRIDOR_OVERRIDE_MARGIN = 6.0
SCORE_SAFEGUARD_OVERRIDE_MARGIN = 8.0
SCORE_SAFEGUARD_BAD_MARGIN = 3.0

_board = HexBoard(R=4, hole_radius=16, spacing=34)
_DIRS  = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

# Target cells are fixed for the match, so fallback evaluation can reuse them
_target_cells = {
    c: set(_board.axial_of_colour(_board.colour_opposites[c]))
    for c in _board.colour_opposites
}

def _configure_tournament_afterstate(agent):
    # Tournament turns need a much lighter search profile than offline evals
    agent.search_depth = TOURNAMENT_AFTERSTATE_DEPTH
    agent.endgame_depth = TOURNAMENT_AFTERSTATE_ENDGAME_DEPTH
    agent.search_width = TOURNAMENT_AFTERSTATE_WIDTH
    agent.endgame_width = TOURNAMENT_AFTERSTATE_ENDGAME_WIDTH
    agent.response_width = TOURNAMENT_AFTERSTATE_RESPONSE_WIDTH
    agent.endgame_response_width = TOURNAMENT_AFTERSTATE_ENDGAME_RESPONSE_WIDTH
    if "cycle" in AFTERSTATE_SEARCH_HYBRID_MODE:
        agent.search_turn_mode = "cycle"
        agent.search_opponent_mode = "leader"
    elif "mix" in AFTERSTATE_SEARCH_HYBRID_MODE:
        agent.search_turn_mode = "duel"
        agent.search_opponent_mode = "leader"
    elif "brs" in AFTERSTATE_SEARCH_HYBRID_MODE:
        agent.search_turn_mode = "duel"
        agent.search_opponent_mode = "first"
    elif "duel" in AFTERSTATE_SEARCH_HYBRID_MODE:
        agent.search_turn_mode = "duel"
        agent.search_opponent_mode = "first"
    return agent

def _search_route_prefix():
    if USE_AFTERSTATE_SEARCH_SEATMIX:
        return f"afterstate+search-seatmix{SEARCH_SETTLE_SWITCH_TURN}"
    if USE_AFTERSTATE_SEARCH_SEATBASE:
        return f"afterstate+search-seatbase{SEARCH_SETTLE_SWITCH_TURN}"
    if USE_AFTERSTATE_SEARCH_4CORRIDOR6SEATBASE:
        return f"afterstate+search-4corridor6seatbase{SEARCH_SETTLE_SWITCH_TURN}"
    if USE_AFTERSTATE_SEARCH_SEATCORRIDOR:
        return f"afterstate+search-seatcorridor{SEARCH_SETTLE_SWITCH_TURN}"
    if USE_AFTERSTATE_SEARCH_CORRIDOR:
        return f"afterstate+search-corridor{SEARCH_SETTLE_SWITCH_TURN}"
    if USE_AFTERSTATE_SEARCH_SETTLE:
        return f"afterstate+search-settle{SEARCH_SETTLE_SWITCH_TURN}"
    if USE_AFTERSTATE_SEARCH_SEATBRS:
        return "afterstate+search-seatbrs"
    if USE_AFTERSTATE_SEARCH_BRS:
        return "afterstate+search-brs"
    return None

def _search_route_display_label():
    if USE_AFTERSTATE_SEARCH_SEATMIX:
        return f"Afterstate Search SeatMix {SEARCH_SETTLE_SWITCH_TURN}"
    if USE_AFTERSTATE_SEARCH_SEATBASE:
        return f"Afterstate Search SeatBase {SEARCH_SETTLE_SWITCH_TURN}"
    if USE_AFTERSTATE_SEARCH_4CORRIDOR6SEATBASE:
        return f"Afterstate Search Corridor/SeatBase {SEARCH_SETTLE_SWITCH_TURN}"
    if USE_AFTERSTATE_SEARCH_SEATCORRIDOR:
        return f"Afterstate Search SeatCorridor {SEARCH_SETTLE_SWITCH_TURN}"
    if USE_AFTERSTATE_SEARCH_CORRIDOR:
        return f"Afterstate Search Corridor {SEARCH_SETTLE_SWITCH_TURN}"
    if USE_AFTERSTATE_SEARCH_SETTLE:
        return f"Afterstate Search Settle {SEARCH_SETTLE_SWITCH_TURN}"
    if USE_AFTERSTATE_SEARCH_SEATBRS:
        return "Afterstate Search SeatBRS"
    if USE_AFTERSTATE_SEARCH_BRS:
        return "Afterstate Search BRS"
    return None

def _locked_route_label():
    return _search_route_display_label() or f"Afterstate ({AFTERSTATE_SEARCH_HYBRID_MODE})"

def _print_tournament_preflight():
    print(f"Route: {_locked_route_label()}")
    required = (
        ("2p model checkpoint", AFTERSTATE_TWOPLAYER_MODEL),
        ("3+p multiplayer checkpoint", AFTERSTATE_MULTIPLAYER_MODEL),
    )
    for label, path in required:
        if os.path.exists(path):
            print(f"  OK: {label}")
        else:
            print(f"  MISSING: {label} -> {path}")

def rpc(payload, timeout_sec=10.0):
    # The tournament server expects one request per short lived socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(float(timeout_sec))
    try:
        s.connect((HOST, PORT))
    except Exception as e:
        try:
            s.close()
        except Exception:
            pass
        return {"ok": False, "error": f"connect-failed: {e}"}
    try:
        s.sendall(json.dumps(payload).encode("utf-8"))
        chunks = []
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
    except Exception as e:
        return {"ok": False, "error": f"rpc-failed: {e}"}
    finally:
        try:
            s.close()
        except Exception:
            pass
    if not data:
        return {"ok": False, "error": "no-response"}
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"bad-json: {e}"}

def _axial_dist(idx1, idx2):
    c1, c2 = _board.cells[idx1], _board.cells[idx2]
    return max(abs(c1.q - c2.q),
               abs(c1.r - c2.r),
               abs((-c1.q - c1.r) - (-c2.q - c2.r)))

def _total_dist(positions, target_cells):
    # Assignment distance prevents unfinished pins from sharing the same target cell
    if not target_cells or not positions:
        return 0

    in_target  = set(p for p in positions if p in target_cells)
    outside    = [p for p in positions if p not in target_cells]

    if not outside:
        return 0

    available = [t for t in target_cells if t not in in_target]
    if not available:
        return 0

    # Match constrained pins first to keep crowded endgames realistic
    outside_sorted = sorted(outside,
                            key=lambda p: min(_axial_dist(p, t) for t in available),
                            reverse=True)
    total     = 0
    remaining = list(available)
    for pin in outside_sorted:
        if not remaining:
            break
        best_t = min(remaining, key=lambda t: _axial_dist(pin, t))
        total += _axial_dist(pin, best_t)
        remaining.remove(best_t)

    return total

def _get_moves(pos_dict, colour):
    occupied = set()
    for positions in pos_dict.values():
        occupied.update(positions)

    valid = []
    for pin_id, start_idx in enumerate(pos_dict.get(colour, [])):
        start_cell = _board.cells[start_idx]
        q0, r0 = start_cell.q, start_cell.r
        possible = set()

        for dq, dr in _DIRS:
            ni = _board.hole_index_of.get((q0 + dq, r0 + dr))
            if ni is not None and ni not in occupied:
                possible.add(ni)

        visited = {start_idx}
        stack   = [start_idx]
        while stack:
            curr = stack.pop()
            cq, cr = _board.cells[curr].q, _board.cells[curr].r
            for dq, dr in _DIRS:
                adj  = _board.hole_index_of.get((cq + dq,     cr + dr))
                land = _board.hole_index_of.get((cq + 2 * dq, cr + 2 * dr))
                if adj is None or land is None:
                    continue
                if adj in occupied and land not in occupied and land not in visited:
                    possible.add(land)
                    visited.add(land)
                    stack.append(land)

        for dest in possible:
            if start_cell.postype != colour and _board.cells[dest].postype == colour:
                continue
            valid.append((pin_id, dest))

    return valid

def _sim(pos_dict, colour, pin_id, dest):
    new = {c: list(p) for c, p in pos_dict.items()}
    new[colour][pin_id] = dest
    return new

def _evaluate(pos_dict, my_colour):
    my_pos    = pos_dict.get(my_colour, [])
    my_target = _target_cells[my_colour]
    my_dist   = _total_dist(my_pos, my_target)
    my_in     = sum(1 for p in my_pos if p in my_target)

    # Opponent blockers matter, but transient blockers should not dominate progress
    opp_blocking = sum(
        1 for c, positions in pos_dict.items()
        if c != my_colour
        for p in positions if p in my_target
    )

    return -my_dist + 15 * my_in - 2 * opp_blocking

def _move_score(pos_dict, colour, pin_id, dest):
    # Order promising moves first minimax still makes the final decision
    positions = pos_dict.get(colour, [])
    target    = _target_cells[colour]
    old_dist  = _total_dist(positions, target)
    simulated = list(positions)
    simulated[pin_id] = dest
    score = old_dist - _total_dist(simulated, target)
    if dest in target and positions[pin_id] not in target:
        score += 20
    if positions[pin_id] in target and dest not in target:
        score -= 30
    return score

def _mean_cartesian(cell_ids):
    points = [_board.cartesian[int(cell_id)] for cell_id in cell_ids]
    if not points:
        return (0.0, 0.0)
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )

def _distance_to_segment(point, start, end):
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

def _corridor_distance(colour, cell_id):
    target_colour = _board.colour_opposites.get(str(colour))
    if not target_colour:
        return 0.0
    home = _board.axial_of_colour(str(colour))
    target = _board.axial_of_colour(target_colour)
    if not home or not target:
        return 0.0
    spacing = float(getattr(_board, "spacing", 34) or 34)
    return _distance_to_segment(
        _board.cartesian[int(cell_id)],
        _mean_cartesian(home),
        _mean_cartesian(target),
    ) / spacing

def _corridor_features(server_state, colour, action, corridor_width=SEARCH_CORRIDOR_WIDTH):
    if action is None:
        return None
    pin_id, dest = action
    pin_id = int(pin_id)
    dest = int(dest)
    positions = server_state.get("pins", {}).get(colour, [])
    if pin_id < 0 or pin_id >= len(positions):
        return None

    origin = int(positions[pin_id])
    targets = _target_cells.get(colour, set())
    try:
        progress = _move_score(server_state.get("pins", {}), colour, pin_id, dest)
    except Exception:
        progress = 0.0

    origin_lane = _corridor_distance(colour, origin)
    dest_lane = _corridor_distance(colour, dest)
    return {
        "origin": origin,
        "dest": dest,
        "progress": float(progress),
        "jump": int(_axial_dist(origin, dest)),
        "origin_lane": float(origin_lane),
        "dest_lane": float(dest_lane),
        "lane_delta": float(dest_lane - origin_lane),
        "outside_dest": max(0.0, float(dest_lane) - float(corridor_width)),
        "enters_target": dest in targets and origin not in targets,
        "stays_target": dest in targets and origin in targets,
    }

def _corridor_score(features):
    if features is None:
        return float("-inf")
    score = features["progress"] * 3.0
    score -= features["outside_dest"] * 4.0
    score -= max(0.0, features["lane_delta"]) * 2.0
    if features["origin_lane"] > SEARCH_CORRIDOR_WIDTH and features["lane_delta"] < 0:
        score += min(8.0, -features["lane_delta"] * 4.0)
    if features["dest_lane"] <= SEARCH_CORRIDOR_WIDTH:
        score += 2.0
    if features["enters_target"]:
        score += 12.0
    elif features["stays_target"]:
        score += 4.0
    if features["jump"] >= 3:
        score += min(8.0, features["jump"] * 1.5)
    return score

def _corridor_adjust_action(
    server_state,
    colour,
    action,
    candidate_actions,
    corridor_width=SEARCH_CORRIDOR_WIDTH,
    override_margin=SEARCH_CORRIDOR_OVERRIDE_MARGIN,
):
    if action is None or not candidate_actions:
        return action
    if USE_AFTERSTATE_SEARCH_SEATCORRIDOR and _search_seatbase_base_trigger(server_state):
        return action

    base_features = _corridor_features(server_state, colour, action, corridor_width=corridor_width)
    base_score = _corridor_score(base_features)
    best_action = action
    best_score = base_score

    for candidate in candidate_actions:
        features = _corridor_features(server_state, colour, candidate, corridor_width=corridor_width)
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
        margin += 8.0

    return best_action if best_score > base_score + margin else action

def _should_use_corridor_guide(server_state):
    if USE_AFTERSTATE_SEARCH_4CORRIDOR6SEATBASE:
        return _player_count(server_state) == 4
    return USE_AFTERSTATE_SEARCH_SEATCORRIDOR or USE_AFTERSTATE_SEARCH_CORRIDOR

def _pins_in_target(positions, target_cells):
    return sum(1 for position in positions if position in target_cells)

def _nearest_target_distance(cell, target_cells):
    if not target_cells:
        return 0
    return min(_axial_dist(int(cell), int(target)) for target in target_cells)

def _score_race_pressure(server_state, colour, my_pins, my_distance):
    try:
        move_count = int(server_state.get("move_count", 0) or 0)
    except (TypeError, ValueError):
        move_count = 0

    player_count = max(1, _player_count(server_state))
    if player_count <= 2:
        midgame_turn, late_turn = 50, 80
    elif player_count == 3:
        midgame_turn, late_turn = 60, 95
    else:
        midgame_turn, late_turn = 72, 110

    pressure = 0.0
    if move_count >= midgame_turn:
        pressure = max(pressure, 0.45)
    if move_count >= late_turn:
        pressure = max(pressure, 0.85)
    if move_count >= late_turn + max(24, 8 * player_count):
        pressure = max(pressure, 1.15)

    my_score_proxy = 100.0 * float(my_pins) - 2.0 * float(my_distance)
    pins = server_state.get("pins", {})
    for opp_colour, opp_positions in pins.items():
        opp_colour = str(opp_colour)
        if opp_colour == str(colour):
            continue
        opp_targets = _target_cells.get(opp_colour, set())
        if not opp_targets:
            continue
        try:
            opp_positions = [int(pos) for pos in opp_positions]
        except (TypeError, ValueError):
            continue
        opp_pins = _pins_in_target(opp_positions, opp_targets)
        opp_distance = _total_dist(opp_positions, opp_targets)
        opp_score_proxy = 100.0 * float(opp_pins) - 2.0 * float(opp_distance)
        lead = opp_score_proxy - my_score_proxy
        if lead > 60.0:
            pressure = max(pressure, min(1.45, lead / 180.0))
        if opp_pins > my_pins:
            pressure = max(pressure, min(1.65, 0.75 + 0.25 * (opp_pins - my_pins)))
        elif opp_pins == my_pins and opp_distance + 8 < my_distance:
            pressure = max(pressure, 0.65)
        if opp_pins >= 8 and opp_pins >= my_pins:
            pressure = max(pressure, 1.15)

    return min(1.8, pressure)

def _score_delta_features(server_state, colour, action):
    # Lightweight score proxy used to keep clock driven moves aligned with server scoring
    if action is None:
        return None
    try:
        pin_id, dest = int(action[0]), int(action[1])
    except (TypeError, ValueError, IndexError):
        return None

    pins = server_state.get("pins", {})
    positions = [int(pos) for pos in pins.get(colour, [])]
    target = _target_cells.get(colour, set())
    if not positions or not target or pin_id < 0 or pin_id >= len(positions):
        return None

    origin = int(positions[pin_id])
    before_distance = _total_dist(positions, target)
    before_pins = _pins_in_target(positions, target)
    race_pressure = _score_race_pressure(server_state, colour, before_pins, before_distance)

    simulated = list(positions)
    simulated[pin_id] = dest
    after_distance = _total_dist(simulated, target)
    after_pins = _pins_in_target(simulated, target)

    pin_gain = after_pins - before_pins
    distance_gain = before_distance - after_distance
    origin_target_distance = _nearest_target_distance(origin, target)
    dest_target_distance = _nearest_target_distance(dest, target)
    pin_distance_gain = origin_target_distance - dest_target_distance
    enters_target = dest in target and origin not in target
    leaves_target = origin in target and dest not in target
    stays_target = origin in target and dest in target

    # Mirror the servers pin and distance scoring without guessing its move bonus
    score = 100.0 * pin_gain + 2.0 * distance_gain
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
        weight = 18.0 if before_pins >= len(positions) - 2 else 10.0
        if origin not in target:
            score += weight * pin_distance_gain
            if pin_distance_gain <= 0 and not enters_target:
                score -= 22.0
        elif stays_target:
            score -= 6.0
    last_pin_pressure = before_pins >= len(positions) - 1
    if last_pin_pressure:
        if origin not in target:
            score += 42.0 * pin_distance_gain
            if before_distance <= 3 and pin_distance_gain > 0:
                score += 80.0
            if pin_distance_gain <= 0 and not enters_target:
                score -= 80.0
        elif not stays_target:
            score -= 120.0
    if race_pressure > 0.0:
        if origin not in target:
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
        jump = int(_axial_dist(origin, dest))
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
        "origin": origin,
        "dest": dest,
    }

def _game_time_limit_seconds(server_state):
    return 60.0 * max(1, _player_count(server_state))

def _elapsed_game_fraction(server_state, game_started_at=None):
    if game_started_at is None:
        return 0.0
    try:
        elapsed = max(0.0, time.time() - float(game_started_at))
    except (TypeError, ValueError):
        return 0.0
    limit = _game_time_limit_seconds(server_state)
    if limit <= 0:
        return 0.0
    return min(2.0, elapsed / limit)

def _sprint_pressure(server_state, colour=None, game_started_at=None):
    player_count = max(1, _player_count(server_state))
    if player_count <= 2:
        return 0.0

    try:
        move_count = int(server_state.get("move_count", 0) or 0)
    except (TypeError, ValueError):
        move_count = 0

    turn_start = TOURNAMENT_SPRINT_TURN_START * player_count
    turn_full = TOURNAMENT_SPRINT_TURN_FULL * player_count
    turn_pressure = 0.0
    if turn_full > turn_start:
        turn_pressure = (float(move_count) - turn_start) / (turn_full - turn_start)

    time_fraction = _elapsed_game_fraction(server_state, game_started_at)
    time_pressure = 0.0
    if TOURNAMENT_SPRINT_TIME_FULL > TOURNAMENT_SPRINT_TIME_START:
        time_pressure = (
            (time_fraction - TOURNAMENT_SPRINT_TIME_START)
            / (TOURNAMENT_SPRINT_TIME_FULL - TOURNAMENT_SPRINT_TIME_START)
        )

    pressure = max(0.0, turn_pressure, time_pressure)
    if colour is not None:
        my_pins = _pins_in_goal(server_state, colour)
        my_dist = _distance_to_goal(server_state, colour)
        if my_pins <= 6 and (move_count >= 16 * player_count or time_fraction >= 0.35):
            pressure += 0.25
        if my_pins <= 8 and (move_count >= 24 * player_count or time_fraction >= 0.50):
            pressure += 0.25
        if my_pins >= 9 and my_dist <= 4:
            pressure += 0.35

    return min(2.0, pressure)

def _sprint_action(server_state, colour, candidate_actions, pressure=1.0):
    if not candidate_actions:
        return None

    best_action = None
    best_score = float("-inf")
    pressure = max(0.0, min(2.0, float(pressure)))

    for candidate in candidate_actions:
        features = _score_delta_features(server_state, colour, candidate)
        if features is None:
            continue

        sprint_score = features["score"]
        sprint_score += 36.0 * max(0.0, features["pin_distance_gain"])
        sprint_score += 70.0 * max(0, features["pin_gain"])
        sprint_score += 18.0 * max(0.0, features["distance_gain"])

        if features["enters_target"]:
            sprint_score += 85.0
        if features["leaves_target"]:
            sprint_score -= 700.0
        if features["finishes_game"]:
            sprint_score += 2000.0

        unfinished = features["pin_count"] - features["pins_before"]
        if unfinished <= 3:
            sprint_score += 45.0 * pressure * max(0.0, features["pin_distance_gain"])
            if features["pin_distance_gain"] <= 0 and not features["enters_target"]:
                sprint_score -= 95.0 * pressure
        elif features["pins_before"] <= 7:
            sprint_score += 24.0 * pressure * max(0.0, features["pin_distance_gain"])

        if features["pin_gain"] <= 0 and features["pin_distance_gain"] <= 0 and not features["enters_target"]:
            sprint_score -= 45.0 * (1.0 + pressure)

        if sprint_score > best_score:
            best_score = sprint_score
            best_action = candidate

    return best_action

def _immediate_finish_action(server_state, colour, candidate_actions):
    finishers = []
    for candidate in candidate_actions:
        features = _score_delta_features(server_state, colour, candidate)
        if features is not None and features["finishes_game"]:
            finishers.append((features["score"], candidate))
    if not finishers:
        return None
    return max(finishers, key=lambda item: item[0])[1]

def _best_legal_fallback_action(server_state, colour, candidate_actions):
    scored = []
    for candidate in candidate_actions:
        features = _score_delta_features(server_state, colour, candidate)
        if features is not None:
            scored.append((features["score"], candidate))
    if scored:
        return max(scored, key=lambda item: item[0])[1]
    return candidate_actions[0] if candidate_actions else None

def _legalize_action(server_state, colour, action, valid_actions, candidate_actions):
    if action is None:
        return action
    try:
        action = (int(action[0]), int(action[1]))
    except (TypeError, ValueError, IndexError):
        action = None
    legal = {(int(pin_id), int(dest)) for pin_id, dest in valid_actions}
    if action in legal:
        return action
    replacement = _best_legal_fallback_action(
        server_state,
        colour,
        candidate_actions if candidate_actions else valid_actions,
    )
    if replacement is not None:
        _decision_log(f"  Legal guard: replaced invalid move with {replacement[0]}->{replacement[1]}")
    return replacement

def _flatten_legal_moves(legal_moves):
    valid_actions = []
    for raw_pin_id, destinations in (legal_moves or {}).items():
        try:
            pin_id = int(raw_pin_id)
        except (TypeError, ValueError):
            continue
        for raw_dest in destinations or []:
            try:
                valid_actions.append((pin_id, int(raw_dest)))
            except (TypeError, ValueError):
                continue
    return valid_actions

def _retry_action_after_rejection(server_state, colour, legal_moves, recent_moves):
    valid_actions = _flatten_legal_moves(legal_moves)
    if not valid_actions:
        return None
    safe_actions = [
        action for action in valid_actions
        if not is_reverse_action(colour, action, recent_moves)
    ]
    candidate_actions = safe_actions if safe_actions else valid_actions
    action = _immediate_finish_action(server_state, colour, valid_actions)
    if action is None:
        action = _best_legal_fallback_action(server_state, colour, candidate_actions)
    return _legalize_action(server_state, colour, action, valid_actions, candidate_actions)

def _score_safeguard_action(server_state, colour, action, candidate_actions):
    if action is None or not candidate_actions:
        return action

    base_features = _score_delta_features(server_state, colour, action)
    if base_features is None:
        return action

    best_action = action
    best_features = base_features
    cleanup_action = None
    cleanup_features = None
    for candidate in candidate_actions:
        features = _score_delta_features(server_state, colour, candidate)
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

    margin = SCORE_SAFEGUARD_BAD_MARGIN if base_features["score"] < 0 else SCORE_SAFEGUARD_OVERRIDE_MARGIN
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
        return cleanup_action

    return best_action if best_features["score"] > base_features["score"] + margin else action

# The fallback search clears this cache every move, so entries stay decision local
_tt: dict = {}

def _pos_key(pos_dict, current_colour, depth, maximizing):
    return (
        hash(tuple(sorted((c, tuple(sorted(p))) for c, p in pos_dict.items()))),
        current_colour, depth, maximizing
    )

def _minimax(pos_dict, my_colour, player_order, turn_idx, depth, alpha, beta, deadline):
    # Paranoid style search maximize own outcome and treat other players as one side
    # Any fully completed home triangle is treated as a terminal result
    for colour, positions in pos_dict.items():
        if len(positions) == 10 and all(p in _target_cells[colour] for p in positions):
            return 10000 if colour == my_colour else -10000

    if depth == 0 or time.time() > deadline:
        return _evaluate(pos_dict, my_colour)

    current_colour = player_order[turn_idx]
    maximizing     = (current_colour == my_colour)
    key            = _pos_key(pos_dict, current_colour, depth, maximizing)
    if key in _tt:
        return _tt[key]

    moves = _get_moves(pos_dict, current_colour)
    next_idx = (turn_idx + 1) % len(player_order)

    if not moves:
        result = _minimax(pos_dict, my_colour, player_order, next_idx, depth - 1, alpha, beta, deadline)
        _tt[key] = result
        return result

    moves.sort(key=lambda m: _move_score(pos_dict, current_colour, m[0], m[1]),
               reverse=maximizing)

    if maximizing:
        best = float('-inf')
        searched = False
        for pin_id, dest in moves:
            if time.time() > deadline:
                break
            searched = True
            score = _minimax(_sim(pos_dict, current_colour, pin_id, dest),
                             my_colour, player_order, next_idx, depth - 1, alpha, beta, deadline)
            best  = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        if not searched:
            best = _evaluate(pos_dict, my_colour)
    else:
        best = float('inf')
        searched = False
        for pin_id, dest in moves:
            if time.time() > deadline:
                break
            searched = True
            score = _minimax(_sim(pos_dict, current_colour, pin_id, dest),
                             my_colour, player_order, next_idx, depth - 1, alpha, beta, deadline)
            best  = min(best, score)
            beta  = min(beta, best)
            if beta <= alpha:
                break
        if not searched:
            best = _evaluate(pos_dict, my_colour)

    _tt[key] = best
    return best

def pick_best_action(server_state, my_colour, legal_moves, depth=4, time_budget=1.2):
    global _tt
    _tt = {}

    pos_dict      = {c: list(p) for c, p in server_state.get("pins", {}).items()}
    valid_actions = [(int(pid), dest) for pid, dests in legal_moves.items() for dest in dests]
    if not valid_actions:
        return None

    # Server turn order is the real authority The two player fallback only
    # exists so local testing still works if that information is missing
    players      = server_state.get("players", [])
    all_colours  = [p["colour"] for p in players if "colour" in p]
    if not all_colours:
        all_colours = [my_colour, _board.colour_opposites.get(my_colour, my_colour)]

    # Rotate the order so the recursive search starts from the controlled colour
    if my_colour in all_colours:
        idx = all_colours.index(my_colour)
        all_colours = all_colours[idx:] + all_colours[:idx]

    # After the controlled move, search continues with the next colour
    next_idx = 1 % len(all_colours)

    best_action = None
    best_score  = float('-inf')
    search_start = time.time()
    deadline    = search_start + float(time_budget)

    for d in range(1, depth + 1):
        if time.time() > deadline:
            break

        candidate_action = None
        candidate_score  = float('-inf')
        alpha = float('-inf')
        beta  = float('inf')

        sorted_actions = sorted(valid_actions,
                                key=lambda m: _move_score(pos_dict, my_colour, m[0], m[1]),
                                reverse=True)

        for pin_id, dest in sorted_actions:
            if time.time() > deadline:
                break
            score = _minimax(_sim(pos_dict, my_colour, pin_id, dest),
                             my_colour, all_colours, next_idx, d - 1, alpha, beta, deadline)
            score += random.uniform(0, 1e-4)
            if score > candidate_score:
                candidate_score  = score
                candidate_action = (pin_id, dest)
            alpha = max(alpha, candidate_score)

        if candidate_action is not None:
            best_action = candidate_action
            best_score  = candidate_score

        elapsed = time.time() - search_start
        _decision_log(
            f"  depth={d}: score={best_score:.1f}  action={best_action}  t={elapsed:.2f}s"
            f"  players={len(all_colours)}"
        )

    return best_action if best_action else valid_actions[0]

def should_use_learned_agent(server_state, my_colour):
    # Use the learned route for supported player counts multiplayer safeguards
    # live inside the afterstate file
    players = server_state.get("players", [])
    player_count = len(players) if players else len(server_state.get("pins", {}))
    opposite = _board.colour_opposites.get(my_colour, "")
    return (
        player_count in SUPPORTED_AFTERSTATE_PLAYER_COUNTS
        and my_colour in AFTERSTATE_TOURNAMENT_COLORS
        and opposite in AFTERSTATE_TOURNAMENT_COLORS
    )

def _choose_fallback_action(state, colour, legal_moves, candidate_actions, sprint_pressure):
    player_count = _player_count(state)
    if player_count not in SUPPORTED_AFTERSTATE_PLAYER_COUNTS:
        # Unsupported sizes get a quick progress first move to protect the clock
        action = _sprint_action(
            state,
            colour,
            candidate_actions,
            max(1.0, sprint_pressure),
        )
        return action or _best_legal_fallback_action(state, colour, candidate_actions)

    fallback_depth = TWO_PLAYER_FALLBACK_DEPTH if player_count == 2 else MULTI_PLAYER_FALLBACK_DEPTH
    fallback_budget = TWO_PLAYER_FALLBACK_TIME_BUDGET if player_count == 2 else MULTI_PLAYER_FALLBACK_TIME_BUDGET
    return pick_best_action(
        state,
        colour,
        legal_moves,
        depth=fallback_depth,
        time_budget=fallback_budget,
    )

def is_reverse_action(current_player, action, recent_moves, lookback=8):
    pin_id, dest = action
    # Recent moves by this pin identify small back and forth cycles
    pin_history = [
        (old_cell, new_cell)
        for colour, old_pin_id, old_cell, new_cell in recent_moves
        if colour == current_player and int(old_pin_id) == int(pin_id)
    ]
    for old_cell, _ in pin_history[-lookback:]:
        if int(old_cell) == int(dest):
            return True
    return False

def remember_move(recent_moves, current_player, pin_id, old_cell, new_cell):
    try:
        entry = (str(current_player), int(pin_id), int(old_cell), int(new_cell))
    except (TypeError, ValueError):
        return
    recent_moves.append(entry)
    if len(recent_moves) > 30:
        recent_moves.pop(0)

class _PinProxy:
    __slots__ = ("id", "color", "axialindex")
    def __init__(self, pin_id, color, axialindex):
        self.id        = pin_id
        self.color     = color
        self.axialindex = axialindex

def _make_pins_on_board(server_state):
    proxies = []
    for colour, indices in server_state.get("pins", {}).items():
        for pin_id, idx in enumerate(indices):
            proxies.append(_PinProxy(pin_id, colour, idx))
    return proxies

def _pins_in_goal(server_state, colour):
    positions = server_state.get("pins", {}).get(colour, [])
    targets = _target_cells.get(colour, set())
    return sum(1 for pos in positions if pos in targets)

def _distance_to_goal(server_state, colour):
    positions = list(server_state.get("pins", {}).get(colour, []))
    return _total_dist(positions, _target_cells.get(colour, set()))

def _leader_metrics(server_state, colour):
    own_colour = str(colour)
    best_colour = None
    best_pins = 0
    best_dist = float("inf")
    best_key = None
    for opponent in server_state.get("pins", {}).keys():
        opponent = str(opponent)
        if opponent == own_colour:
            continue
        pins = _pins_in_goal(server_state, opponent)
        dist = _distance_to_goal(server_state, opponent)
        key = (pins, -dist)
        if best_key is None or key > best_key:
            best_key = key
            best_colour = opponent
            best_pins = pins
            best_dist = dist
    return best_colour, best_pins, best_dist

def _afterstate_time_budget(server_state, colour, game_started_at=None):
    # Tournament play needs a predictable clock profile extra time is reserved
    # for close races or leaders near home
    sprint_pressure = _sprint_pressure(server_state, colour, game_started_at)
    if sprint_pressure >= TOURNAMENT_HARD_SPRINT_TRIGGER:
        return TOURNAMENT_AFTERSTATE_CRUISE_BUDGET

    my_pins = _pins_in_goal(server_state, colour)
    my_dist = _distance_to_goal(server_state, colour)
    _, opp_pins, opp_dist = _leader_metrics(server_state, colour)
    race_margin = 18.0 * (my_pins - opp_pins) + 0.35 * (opp_dist - my_dist)

    if opp_pins >= 8 or opp_dist <= 14 or abs(race_margin) <= 10.0:
        return TOURNAMENT_AFTERSTATE_PANIC_BUDGET

    if my_pins >= 6 and race_margin >= 28.0 and my_dist + 8 < opp_dist:
        return TOURNAMENT_AFTERSTATE_CRUISE_BUDGET

    if sprint_pressure >= TOURNAMENT_SPRINT_TRIGGER:
        return min(TOURNAMENT_AFTERSTATE_BASE_BUDGET, 0.16)

    return TOURNAMENT_AFTERSTATE_BASE_BUDGET

def _progress_bar(count, total=10):
    count = max(0, min(int(count), total))
    return "#" * count + "." * (total - count)

def _state_players(server_state):
    players = server_state.get("players", [])
    if players:
        return players
    return [
        {"name": colour, "colour": colour}
        for colour in server_state.get("pins", {}).keys()
    ]

def _render_terminal_dashboard(server_state, my_colour, my_name, rl_agent_desc, game_id):
    # This keeps the tournament client readable in a plain terminal without
    # needing the local Tk GUI
    current_turn = server_state.get("current_turn_colour", "-")
    status = server_state.get("status", "-")
    move_count = server_state.get("move_count", 0)
    last_move = server_state.get("last_move")

    print("\033[2J\033[H", end="")
    print("=" * 76)
    print(f"CHINESE CHECKERS TOURNAMENT  |  Game {game_id}  |  Status {status}  |  Move {move_count}")
    print(f"You: {my_name} ({my_colour.upper()})  |  Client: {rl_agent_desc}")
    print(f"Current turn: {str(current_turn).upper()}")
    print("-" * 76)
    print("Players:")

    for player in _state_players(server_state):
        colour = str(player.get("colour", ""))
        name = str(player.get("name", colour))
        home = _pins_in_goal(server_state, colour)
        dist = _distance_to_goal(server_state, colour)
        marker = "YOU" if colour == my_colour else ""
        turn_marker = "*" if colour == current_turn else " "
        print(
            f" {turn_marker} {colour.upper():11} {name[:18]:18} "
            f"[{_progress_bar(home)}] home={home}/10 dist={dist:>3} {marker}"
        )

    if last_move:
        print("-" * 76)
        print(
            f"Last move: {last_move['by']} ({last_move['colour']}) "
            f"{last_move['from']}->{last_move['to']}  [{last_move['move_ms']:.1f}ms]"
        )

    print("-" * 76)
    print("Board:")
    _board.print_ascii(pins=_make_pins_on_board(server_state), empty=".")
    print("=" * 76)
    sys.stdout.flush()

def _render_compact_status(server_state, my_colour):
    current_turn = str(server_state.get("current_turn_colour", "-"))
    status = str(server_state.get("status", "-"))
    move_count = server_state.get("move_count", 0)
    home = _pins_in_goal(server_state, my_colour)
    dist = _distance_to_goal(server_state, my_colour)
    print(
        f"Move {move_count}: status={status} turn={current_turn.upper()} "
        f"home={home}/10 dist={dist}"
    )

def _player_count(server_state):
    players = server_state.get("players", [])
    return len(players) if players else len(server_state.get("pins", {}))

def _search_hybrid_early_trigger(server_state, turn_limit=None, min_players=None):
    min_player_count = 4 if min_players is None else int(min_players)
    if _player_count(server_state) < min_player_count:
        return False
    try:
        move_count = int(server_state.get("move_count", 0) or 0)
    except (TypeError, ValueError):
        move_count = 0
    limit = SEARCH_EARLY_TURN_LIMIT if turn_limit is None else int(turn_limit)
    return move_count < limit

def _current_player_seat(server_state, colour=None):
    colour = str(colour or server_state.get("current_turn_colour", ""))
    players = server_state.get("players", []) or []
    for index, player in enumerate(players):
        if str(player.get("colour")) == colour:
            return index
    return None

def _search_seatbase_base_trigger(server_state):
    if not (
        USE_AFTERSTATE_SEARCH_SEATBASE
        or USE_AFTERSTATE_SEARCH_4CORRIDOR6SEATBASE
        or USE_AFTERSTATE_SEARCH_SEATCORRIDOR
        or USE_AFTERSTATE_SEARCH_SEATBRS
        or USE_AFTERSTATE_SEARCH_SEATMIX
    ):
        return False
    if _player_count(server_state) != 6:
        return False
    colour = str(server_state.get("current_turn_colour", ""))
    seat = _current_player_seat(server_state, colour)
    return seat in SEARCH_SEATBASE_BASE_SEATS or colour in SEARCH_SEATBASE_BASE_COLOURS

def _should_use_best_reply_pressure(server_state):
    if not USE_AFTERSTATE_SEARCH_SEATMIX:
        return False
    if _search_seatbase_base_trigger(server_state):
        return False
    if _player_count(server_state) < 4:
        return False
    try:
        move_count = int(server_state.get("move_count", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        move_count = 0
    if move_count < SEARCH_SETTLE_SWITCH_TURN:
        return False

    players = _state_players(server_state)
    colours = [str(player.get("colour")) for player in players if player.get("colour") is not None]
    current = str(server_state.get("current_turn_colour", ""))
    if current not in colours:
        return False

    leader = _best_reply_target_colour(server_state, current, colours)
    if leader is None:
        return False

    my_pins = _pins_in_goal(server_state, current)
    leader_pins = _pins_in_goal(server_state, leader)
    my_dist = _distance_to_goal(server_state, current)
    leader_dist = _distance_to_goal(server_state, leader)
    return (
        leader_pins >= my_pins + SEARCH_SEATMIX_LEADER_PIN_GAP
        or leader_dist <= my_dist - SEARCH_SEATMIX_LEADER_DIST_GAP
    )

def _search_mode_for_state(server_state):
    if USE_AFTERSTATE_SEARCH_BRS or USE_AFTERSTATE_SEARCH_SEATBRS:
        return "duel", "first"
    if USE_AFTERSTATE_SEARCH_SEATMIX and _should_use_best_reply_pressure(server_state):
        return "duel", "first"
    if (
        USE_AFTERSTATE_SEARCH_SETTLE
        or USE_AFTERSTATE_SEARCH_SEATBASE
        or USE_AFTERSTATE_SEARCH_4CORRIDOR6SEATBASE
        or USE_AFTERSTATE_SEARCH_SEATCORRIDOR
        or USE_AFTERSTATE_SEARCH_CORRIDOR
        or USE_AFTERSTATE_SEARCH_SEATMIX
    ):
        try:
            move_count = int(server_state.get("move_count", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            move_count = 0
        if move_count >= SEARCH_SETTLE_SWITCH_TURN:
            return "cycle", "leader"
        return "duel", "first"
    return None

def _apply_search_mode_for_state(agent, server_state):
    mode = _search_mode_for_state(server_state)
    if agent is None or mode is None or not hasattr(agent, "search_turn_mode"):
        return
    agent.search_turn_mode, agent.search_opponent_mode = mode

def _should_use_search_hybrid(server_state):
    if USE_AFTERSTATE_SEARCH_SEATBRS:
        if _search_seatbase_base_trigger(server_state):
            return False
        return _player_count(server_state) >= 4
    if USE_AFTERSTATE_SEARCH_BRS:
        return _player_count(server_state) >= 4
    if USE_AFTERSTATE_SEARCH_SEATMIX:
        if _player_count(server_state) == 2:
            return True
        if _search_seatbase_base_trigger(server_state):
            return False
        return _search_hybrid_early_trigger(server_state, min_players=4)
    if USE_AFTERSTATE_SEARCH_SEATBASE:
        if _player_count(server_state) == 2:
            return True
        if _search_seatbase_base_trigger(server_state):
            return False
        return _search_hybrid_early_trigger(server_state, min_players=4)
    if USE_AFTERSTATE_SEARCH_4CORRIDOR6SEATBASE:
        if _player_count(server_state) == 4:
            return _search_hybrid_early_trigger(server_state, min_players=4)
        if _search_seatbase_base_trigger(server_state):
            return False
        return _search_hybrid_early_trigger(server_state, min_players=4)
    if USE_AFTERSTATE_SEARCH_SEATCORRIDOR:
        if _search_seatbase_base_trigger(server_state):
            return False
        return _search_hybrid_early_trigger(server_state, min_players=4)
    if USE_AFTERSTATE_SEARCH_CORRIDOR:
        return _search_hybrid_early_trigger(server_state, min_players=4)
    if USE_AFTERSTATE_SEARCH_SETTLE:
        return _search_hybrid_early_trigger(server_state, min_players=4)
    return False

def _best_reply_target_colour(server_state, current_colour, colours):
    opponents = [str(colour) for colour in colours if str(colour) != str(current_colour)]
    if not opponents:
        return None
    return max(
        opponents,
        key=lambda colour: (
            _pins_in_goal(server_state, colour),
            -_distance_to_goal(server_state, colour),
        ),
    )

def _search_player_order_for_state(server_state, player_order=None):
    if player_order:
        colours = [str(colour) for colour in player_order]
    else:
        colours = [str(player.get("colour")) for player in _state_players(server_state)]

    current = str(server_state.get("current_turn_colour", ""))
    if current in colours:
        current_index = colours.index(current)
        colours = colours[current_index:] + colours[:current_index]

    if not colours:
        return None

    if not (
        USE_AFTERSTATE_SEARCH_BRS
        or USE_AFTERSTATE_SEARCH_SEATBRS
        or (USE_AFTERSTATE_SEARCH_SEATMIX and _should_use_best_reply_pressure(server_state))
    ):
        return colours
    if len(colours) < 3:
        return colours

    leader = _best_reply_target_colour(server_state, current, colours)
    if leader is None:
        return colours
    return [current, leader] + [
        colour for colour in colours
        if colour not in {current, leader}
    ]

def _select_afterstate_agent(server_state, base_agent, multiplayer_agent, base_search_agent=None, multiplayer_search_agent=None):
    # Route selector two player model, multiplayer model, optional search wrapper
    player_count = _player_count(server_state)
    if _search_seatbase_base_trigger(server_state):
        return base_agent
    if _should_use_search_hybrid(server_state):
        if player_count > 2 and multiplayer_search_agent is not None:
            return multiplayer_search_agent
        if player_count <= 2 and base_search_agent is not None:
            return base_search_agent
        # If search is unavailable, use the same specialist route without search
        if player_count > 2 and multiplayer_agent is not None:
            return multiplayer_agent
        return base_agent
    if player_count > 2 and multiplayer_agent is not None:
        return multiplayer_agent
    return base_agent

def try_load_afterstate(model_candidates=None, agent_name="TournamentAfterstate", search_hybrid=None):
    try:
        from src.afterstate import AfterstateSearchAgent, AfterstateValueAgent
        candidates = model_candidates or (
            AFTERSTATE_TWOPLAYER_MODEL,
            AFTERSTATE_TRAINED_MODEL,
            AFTERSTATE_FINAL_MODEL,
        )
        use_search_hybrid = False if search_hybrid is None else bool(search_hybrid)
        model_path = first_existing(*candidates)
        if os.path.exists(model_path):
            agent_cls = AfterstateSearchAgent if use_search_hybrid else AfterstateValueAgent
            agent = agent_cls(
                state_size=STATE_SIZE,
                player_color="yellow",
                name=f"{agent_name}Search" if use_search_hybrid else agent_name,
            )
            if agent.load_model(str(model_path), verbose=False):
                agent.epsilon = 0.0
                if use_search_hybrid:
                    _configure_tournament_afterstate(agent)
                return agent

            search_agent = AfterstateSearchAgent(
                state_size=STATE_SIZE,
                player_color="yellow",
                name=f"{agent_name}Search",
            )
            if search_agent.load_model(str(model_path), verbose=False):
                search_agent.epsilon = 0.0
                _configure_tournament_afterstate(search_agent)
                return search_agent
        return None
    except Exception as e:
        print(f"Afterstate route unavailable: {e}")
        return None

def main():
    print("Chinese Checkers tournament client")
    _print_tournament_preflight()
    name = input("Enter name: ").strip()
    if not name:
        return

    afterstate_agent = try_load_afterstate()
    multiplayer_afterstate_agent = try_load_afterstate(
        (AFTERSTATE_MULTIPLAYER_MODEL,),
        "TournamentAfterstateMP",
    )
    afterstate_search_agent = None
    multiplayer_afterstate_search_agent = None
    if (
        USE_AFTERSTATE_SEARCH_SEATBASE
        or USE_AFTERSTATE_SEARCH_4CORRIDOR6SEATBASE
        or USE_AFTERSTATE_SEARCH_SEATCORRIDOR
        or USE_AFTERSTATE_SEARCH_CORRIDOR
        or USE_AFTERSTATE_SEARCH_SETTLE
        or USE_AFTERSTATE_SEARCH_SEATMIX
        or USE_AFTERSTATE_SEARCH_BRS
        or USE_AFTERSTATE_SEARCH_SEATBRS
    ):
        afterstate_search_agent = try_load_afterstate(
            agent_name="TournamentAfterstateSearch",
            search_hybrid=True,
        )
        multiplayer_afterstate_search_agent = try_load_afterstate(
            (AFTERSTATE_MULTIPLAYER_MODEL,),
            "TournamentAfterstateMPSearch",
            search_hybrid=True,
        )
    if afterstate_agent is not None or multiplayer_afterstate_agent is not None:
        if multiplayer_afterstate_agent is not None and afterstate_agent is not None:
            prefix = _locked_route_label()
            rl_agent_desc = f"{prefix} (2p model, 3+p multiplayer model)"
        else:
            active_agent = multiplayer_afterstate_agent or afterstate_agent
            rl_agent_desc = f"Afterstate ({active_agent.name})"
    elif afterstate_agent is not None:
        rl_agent_desc = f"Afterstate ({afterstate_agent.name})"
    else:
        rl_agent_desc = "minimax fallback"

    r = rpc({"op": "join", "player_name": name})
    if not r.get("ok"):
        print("Join failed:", r.get("error"))
        return

    game_id   = r["game_id"]
    player_id = r["player_id"]
    colour    = r["colour"]
    print(f"Joined game {game_id} as {colour}  |  Controller: {rl_agent_desc}")

    waiting_printed = False
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") in ("READY_TO_START", "PLAYING"):
            break
        if not waiting_printed:
            print("Waiting for players...")
            waiting_printed = True
        time.sleep(0.5)

    print("Waiting for game start...")
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") == "PLAYING":
            break
        time.sleep(0.5)

    print("Game started\n")
    game_started_at = time.time()

    timeoutnotice_move = -1
    recent_moves       = []
    last_dashboard_key = None

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            print("Error:", st.get("error"))
            return

        state = st["state"]
        dashboard_key = (
            state.get("status"),
            state.get("move_count", 0),
            state.get("current_turn_colour"),
            state.get("turn_timeout_notice"),
        )
        if dashboard_key != last_dashboard_key:
            if TOURNAMENT_VERBOSE_DASHBOARD:
                _render_terminal_dashboard(state, colour, name, rl_agent_desc, game_id)
            elif state.get("current_turn_colour") == colour or state.get("status") == "FINISHED":
                _render_compact_status(state, colour)
            last_dashboard_key = dashboard_key

        if state.get("turn_timeout_notice") and timeoutnotice_move < state.get("move_count", 0):
            print("Timeout notice:", state["turn_timeout_notice"])
            timeoutnotice_move = state.get("move_count", 0)

        if state["status"] == "FINISHED":
            print("\nGame finished")
            for pl in state["players"]:
                sc = pl.get("score")
                if sc:
                    print(f"  {pl['name']} ({pl['colour']}): "
                          f"{sc['final_score']:.1f} "
                          f"[time={sc['time_score']:.1f}, "
                          f"moves({sc['moves']})={sc['move_score']:.1f}, "
                          f"pins={sc['pin_goal_score']:.1f}, "
                          f"dist={sc['distance_score']:.1f}]")
            break

        if state.get("current_turn_colour") == colour and state["status"] == "PLAYING":
            print("\nMy turn")

            legal_req = rpc({"op": "get_legal_moves", "game_id": game_id, "player_id": player_id})
            if not legal_req.get("ok"):
                print("Error getting legal moves:", legal_req.get("error"))
                time.sleep(TOURNAMENT_POLL_SLEEP_SECONDS)
                continue

            legal_moves = legal_req.get("legal_moves", {})
            if not legal_moves:
                print("No legal moves.")
                time.sleep(TOURNAMENT_POLL_SLEEP_SECONDS)
                continue

            valid_actions = _flatten_legal_moves(legal_moves)
            if not valid_actions:
                print("No usable legal moves.")
                time.sleep(TOURNAMENT_POLL_SLEEP_SECONDS)
                continue
            safe_actions = [
                action for action in valid_actions
                if not is_reverse_action(colour, action, recent_moves)
            ]
            candidate_actions = safe_actions if safe_actions else valid_actions

            filtered_legal_moves = {}
            for pin_id, dest in candidate_actions:
                filtered_legal_moves.setdefault(str(pin_id), []).append(dest)

            use_learned_agent = should_use_learned_agent(state, colour)
            learned_actions = valid_actions
            player_order = [
                str(player.get("colour"))
                for player in state.get("players", [])
                if player.get("colour") is not None
            ]
            player_order = _search_player_order_for_state(state, player_order)
            sprint_pressure = _sprint_pressure(state, colour, game_started_at)

            # Decision order is layered instant win, clock sprint, learned route,
            # then legal fallback Learned controllers see the full legal set
            action = None
            action = _immediate_finish_action(state, colour, valid_actions)
            if action is not None:
                _decision_log(f"  Immediate finish: {action[0]}->{action[1]}")
            if action is None and sprint_pressure >= TOURNAMENT_SPRINT_TRIGGER:
                action = _sprint_action(state, colour, candidate_actions, sprint_pressure)
                if action is not None:
                    elapsed = time.time() - game_started_at
                    _decision_log(
                        f"  Sprint scorer: {action[0]}->{action[1]} "
                        f"pressure={sprint_pressure:.2f} elapsed={elapsed:.0f}/"
                        f"{_game_time_limit_seconds(state):.0f}s"
                    )

            active_afterstate_agent = _select_afterstate_agent(
                state,
                afterstate_agent,
                multiplayer_afterstate_agent,
                afterstate_search_agent,
                multiplayer_afterstate_search_agent,
            )
            if action is None and use_learned_agent and active_afterstate_agent is not None:
                try:
                    pins_proxy = _make_pins_on_board(state)
                    time_budget = _afterstate_time_budget(state, colour, game_started_at)
                    mode_label = (
                        "Afterstate+SeatBase"
                        if (
                            USE_AFTERSTATE_SEARCH_SEATBASE
                            or USE_AFTERSTATE_SEARCH_4CORRIDOR6SEATBASE
                            or USE_AFTERSTATE_SEARCH_SEATCORRIDOR
                            or USE_AFTERSTATE_SEARCH_SEATBRS
                            or USE_AFTERSTATE_SEARCH_SEATMIX
                        ) and _search_seatbase_base_trigger(state)
                        else
                        "Afterstate+Search4Corridor6SeatBase"
                        if _should_use_search_hybrid(state) and USE_AFTERSTATE_SEARCH_4CORRIDOR6SEATBASE
                        else
                        "Afterstate+SearchSeatCorridor"
                        if _should_use_search_hybrid(state) and USE_AFTERSTATE_SEARCH_SEATCORRIDOR
                        else
                        "Afterstate+SearchCorridor"
                        if _should_use_search_hybrid(state) and USE_AFTERSTATE_SEARCH_CORRIDOR
                        else
                        "Afterstate+SearchSeatMix"
                        if _should_use_search_hybrid(state) and USE_AFTERSTATE_SEARCH_SEATMIX
                        else
                        "Afterstate+SearchSeatBRS"
                        if _should_use_search_hybrid(state) and USE_AFTERSTATE_SEARCH_SEATBRS
                        else
                        "Afterstate+SearchBRS"
                        if _should_use_search_hybrid(state) and USE_AFTERSTATE_SEARCH_BRS
                        else
                        "Afterstate+SearchSeatBase"
                        if _should_use_search_hybrid(state) and USE_AFTERSTATE_SEARCH_SEATBASE
                        else
                        "Afterstate+SearchSettle"
                        if _should_use_search_hybrid(state) and USE_AFTERSTATE_SEARCH_SETTLE
                        else
                        "Afterstate"
                    )
                    _decision_log(f"  {mode_label} budget: {time_budget:.2f}s")
                    _apply_search_mode_for_state(active_afterstate_agent, state)
                    action = active_afterstate_agent.choose_action_from_board(
                        pins_proxy,
                        colour,
                        learned_actions,
                        _board,
                        time_budget_seconds=time_budget,
                        turn_count=state.get("move_count", 0),
                        move_history=recent_moves,
                        player_order=player_order,
                    )
                except Exception as e:
                    print(f"Afterstate error: {e}")

            if action is None:
                action = _choose_fallback_action(
                    state,
                    colour,
                    filtered_legal_moves if filtered_legal_moves else legal_moves,
                    candidate_actions,
                    sprint_pressure,
                )

            # Last resort legal move to avoid forfeiting the turn
            if action is None:
                fallback_moves = filtered_legal_moves if filtered_legal_moves else legal_moves
                pid_str = next(iter(fallback_moves))
                action  = (int(pid_str), fallback_moves[pid_str][0])

            if _should_use_corridor_guide(state):
                adjusted_action = _corridor_adjust_action(state, colour, action, candidate_actions)
                if adjusted_action != action:
                    _decision_log(
                        f"  Corridor guide: {action[0]}->{action[1]} "
                        f"changed to {adjusted_action[0]}->{adjusted_action[1]}"
                    )
                action = adjusted_action

            adjusted_action = _score_safeguard_action(state, colour, action, candidate_actions)
            if adjusted_action != action:
                _decision_log(
                    f"  Score safeguard: {action[0]}->{action[1]} "
                    f"changed to {adjusted_action[0]}->{adjusted_action[1]}"
                )
            action = adjusted_action
            action = _legalize_action(state, colour, action, valid_actions, candidate_actions)

            pin_id, to_index = action
            print(f"Playing pin {pin_id} -> cell {to_index}")
            try:
                old_cell = state.get("pins", {}).get(colour, [])[int(pin_id)]
            except (TypeError, ValueError, IndexError):
                old_cell = None

            mv = rpc({"op": "move", "game_id": game_id, "player_id": player_id,
                      "pin_id": pin_id, "to_index": to_index})

            if not mv.get("ok"):
                print("Move rejected:", mv.get("error"))
                retry_state_resp = rpc({"op": "get_state", "game_id": game_id})
                retry_state = retry_state_resp.get("state", state) if retry_state_resp.get("ok") else state
                if retry_state.get("current_turn_colour") == colour and retry_state.get("status") == "PLAYING":
                    retry_legal_req = rpc({"op": "get_legal_moves", "game_id": game_id, "player_id": player_id})
                    if retry_legal_req.get("ok"):
                        retry_action = _retry_action_after_rejection(
                            retry_state,
                            colour,
                            retry_legal_req.get("legal_moves", {}),
                            recent_moves,
                        )
                        if retry_action is not None:
                            retry_pin_id, retry_to_index = retry_action
                            try:
                                retry_old_cell = retry_state.get("pins", {}).get(colour, [])[int(retry_pin_id)]
                            except (TypeError, ValueError, IndexError):
                                retry_old_cell = None
                            print(f"Retrying legal move {retry_pin_id} -> cell {retry_to_index}")
                            retry_mv = rpc({
                                "op": "move",
                                "game_id": game_id,
                                "player_id": player_id,
                                "pin_id": retry_pin_id,
                                "to_index": retry_to_index,
                            })
                            if retry_mv.get("ok"):
                                pin_id, to_index, old_cell = retry_pin_id, retry_to_index, retry_old_cell
                                mv = retry_mv
                            else:
                                print("Retry rejected:", retry_mv.get("error"))
                    else:
                        print("Retry skipped: could not refresh legal moves:", retry_legal_req.get("error"))

            if not mv.get("ok"):
                pass
            elif mv.get("status") == "WIN":
                remember_move(recent_moves, colour, pin_id, old_cell, to_index)
                print("YOU WIN!", mv.get("msg"))
            elif mv.get("status") == "DRAW":
                remember_move(recent_moves, colour, pin_id, old_cell, to_index)
                print("DRAW", mv.get("msg"))
            else:
                remember_move(recent_moves, colour, pin_id, old_cell, to_index)

        time.sleep(TOURNAMENT_POLL_SLEEP_SECONDS)

if __name__ == "__main__":
    main()

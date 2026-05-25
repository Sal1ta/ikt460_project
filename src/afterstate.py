# Afterstate agent and its small search extension

import random
import time
from collections import deque

import torch
import torch.nn as nn

from src.board import HexBoard, HEX_DIRECTIONS
from src.network import AfterstateValueNetwork, resolve_torch_device
from src.perspective import REFERENCE_COLOR, indices_to_reference_perspective
from src.searchstate import SearchBoardState

DIRECTIONS = HEX_DIRECTIONS

_REFERENCE_COLOR = REFERENCE_COLOR

def _choose_from_top_actions(scored_actions, top_k=3):
    if not scored_actions:
        return None

    # Exploration stays close to the best few moves instead of wandering over
    # the whole action list
    ranked = sorted(scored_actions, key=lambda item: item[0], reverse=True)
    frontier = ranked[: max(1, min(int(top_k), len(ranked)))]
    best_score = frontier[0][0]
    close_actions = [action for score, action in frontier if score >= best_score - 0.15]
    pool = close_actions or [action for _, action in frontier]
    return random.choice(pool)

def _normalise_colour(colour):
    return str(colour)

def _target_positions(board, player_color):
    target_colour = board.colour_opposites.get(_normalise_colour(player_color), "")
    return set(board.axial_of_colour(target_colour))

def _axial_distance(board, idx1, idx2):
    c1, c2 = board.cells[int(idx1)], board.cells[int(idx2)]
    return max(
        abs(c1.q - c2.q),
        abs(c1.r - c2.r),
        abs((-c1.q - c1.r) - (-c2.q - c2.r)),
    )

def _assignment_distance(board, positions, target_positions):
    if not positions or not target_positions:
        return 0

    target_positions = set(int(t) for t in target_positions)
    in_target = {int(pos) for pos in positions if int(pos) in target_positions}
    outside = [int(pos) for pos in positions if int(pos) not in target_positions]
    available = [t for t in target_positions if t not in in_target]
    if not outside or not available:
        return 0

    outside.sort(
        key=lambda pos: min(_axial_distance(board, pos, target) for target in available),
        reverse=True,
    )

    total = 0
    remaining = list(available)
    for pos in outside:
        if not remaining:
            break
        target = min(remaining, key=lambda t: _axial_distance(board, pos, t))
        total += _axial_distance(board, pos, target)
        remaining.remove(target)
    return total

def _squared_assignment_distance(board, positions, target_positions):
    if not positions or not target_positions:
        return 0

    target_positions = set(int(t) for t in target_positions)
    in_target = {int(pos) for pos in positions if int(pos) in target_positions}
    outside = [int(pos) for pos in positions if int(pos) not in target_positions]
    available = [t for t in target_positions if t not in in_target]
    if not outside or not available:
        return 0

    outside.sort(
        key=lambda pos: min(_axial_distance(board, pos, target) for target in available),
        reverse=True,
    )

    total = 0
    remaining = list(available)
    for pos in outside:
        if not remaining:
            break
        target = min(remaining, key=lambda t: _axial_distance(board, pos, t))
        distance = _axial_distance(board, pos, target)
        total += distance * distance
        remaining.remove(target)
    return total

def _position_state_from_pins(pins_on_board, player_color):
    player = _normalise_colour(player_color)
    # Replay only needs board indices for my pins and their pins
    own_positions = sorted(
        pin.axialindex for pin in pins_on_board if _normalise_colour(pin.color) == player
    )
    opp_positions = sorted(
        pin.axialindex for pin in pins_on_board if _normalise_colour(pin.color) != player
    )
    return tuple(own_positions), tuple(opp_positions)

def _reference_position_state(own_positions, opp_positions, board, player_color):
    colour = _normalise_colour(player_color)
    if colour == _REFERENCE_COLOR:
        return tuple(int(p) for p in own_positions), tuple(int(p) for p in opp_positions)
    return (
        indices_to_reference_perspective(board, colour, own_positions),
        indices_to_reference_perspective(board, colour, opp_positions),
    )

def _encode_from_positions(own_positions, opp_positions, board, player_color):
    # The network only learns one lane orientation Every other lane is rotated
    # into that reference frame before features are built
    if _normalise_colour(player_color) != _REFERENCE_COLOR:
        own_positions, opp_positions = _reference_position_state(
            own_positions,
            opp_positions,
            board,
            player_color,
        )
        player_color = _REFERENCE_COLOR

    board_size = len(board.cells)
    own_layer = [0] * board_size
    opp_layer = [0] * board_size

    for idx in own_positions:
        if 0 <= int(idx) < board_size:
            own_layer[int(idx)] = 1

    for idx in opp_positions:
        if 0 <= int(idx) < board_size:
            opp_layer[int(idx)] = 1

    targets = _target_positions(board, player_color)
    target_layer = [1 if idx in targets else 0 for idx in range(board_size)]
    return own_layer + opp_layer + target_layer

def encode_afterstate(pins_on_board, player_color, board, action):
    # An afterstate is the position right after the controlled move lands
    own_positions, opp_positions = _position_state_from_pins(pins_on_board, player_color)
    move_pin_id, destination = action
    destination = int(destination)

    moved_positions = list(own_positions)
    origin = None

    for pin in pins_on_board:
        if _normalise_colour(pin.color) != _normalise_colour(player_color):
            continue
        if int(pin.id) == int(move_pin_id):
            origin = int(pin.axialindex)
            break

    if origin is None:
        return _encode_from_positions(own_positions, opp_positions, board, player_color)

    for index, position in enumerate(moved_positions):
        if int(position) == origin:
            moved_positions[index] = destination
            break

    moved_positions.sort()
    return _encode_from_positions(tuple(moved_positions), opp_positions, board, player_color)

def _legal_destinations(start_idx, occupied, board, player_color):
    # This rebuilds legal moves from plain position data, which is why replay
    # does not need to store whole environments
    current_cell = board.cells[int(start_idx)]
    q0, r0 = current_cell.q, current_cell.r
    possible = set()

    for dq, dr in DIRECTIONS:
        neighbour = board.hole_index_of.get((q0 + dq, r0 + dr))
        if neighbour is None or neighbour in occupied:
            continue
        destination = board.cells[neighbour]
        if current_cell.postype != _normalise_colour(player_color) and destination.postype == _normalise_colour(player_color):
            continue
        possible.add(neighbour)

    visited = {int(start_idx)}
    stack = [int(start_idx)]

    while stack:
        current = stack.pop()
        cell = board.cells[current]
        for dq, dr in DIRECTIONS:
            adjacent = board.hole_index_of.get((cell.q + dq, cell.r + dr))
            landing = board.hole_index_of.get((cell.q + 2 * dq, cell.r + 2 * dr))
            if adjacent is None or landing is None:
                continue
            if adjacent not in occupied or landing in occupied or landing in visited:
                continue
            destination = board.cells[landing]
            if current_cell.postype != _normalise_colour(player_color) and destination.postype == _normalise_colour(player_color):
                continue
            visited.add(landing)
            possible.add(landing)
            stack.append(landing)

    return sorted(possible)

def generate_afterstates_from_position_state(position_state, board, player_color):
    own_positions, opp_positions = position_state
    own_positions = tuple(sorted(int(pos) for pos in own_positions))
    opp_positions = tuple(sorted(int(pos) for pos in opp_positions))
    occupied = set(own_positions) | set(opp_positions)

    afterstates = []
    seen = set()

    # Different moves can land in the same afterstate, so one copy is enough
    for index, start_idx in enumerate(own_positions):
        legal_moves = _legal_destinations(start_idx, occupied, board, player_color)
        for destination in legal_moves:
            moved_positions = list(own_positions)
            moved_positions[index] = int(destination)
            moved_positions.sort()
            moved_key = tuple(moved_positions)
            if moved_key in seen:
                continue
            seen.add(moved_key)
            afterstates.append(
                _encode_from_positions(moved_key, opp_positions, board, player_color)
            )

    return afterstates

class AfterstateValueAgent:

    def __init__(self, state_size, player_color="yellow", name="AfterstateValueAgent"):
        self.name = name
        self.state_size = state_size
        self.player_color = _normalise_colour(player_color)

        self.learning_rate = 0.0001
        self.gamma = 0.99
        self.n_step = 3
        self.epsilon = 0.15
        self.epsilon_decay = 0.9997
        self.epsilon_min = 0.01
        self.batch_size = 64

        self.device = resolve_torch_device()

        self.value_network = AfterstateValueNetwork(self.state_size).to(self.device)
        self.target_network = AfterstateValueNetwork(self.state_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.learning_rate)

        self.memory = deque(maxlen=30000)
        self.demo_memory = deque(maxlen=30000)
        self.pending_experiences = deque()
        self.demo_fraction = 0.50
        self.train_updates = 0

        # Replay only stores compact position data A fixed helper board lets us
        # rebuild the legal follow up afterstates later without storing full envs
        self.replay_board = HexBoard(R=4, hole_radius=16, spacing=34)
        self.dense_multiplayer_leader_gap = 15.0
        self.dense_multiplayer_jump_weight = 0.0
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.value_network.state_dict())

    def afterstate_from_env_action(self, env, action, player_color=None):
        colour = _normalise_colour(player_color or env.get_current_player())
        return encode_afterstate(env.pins_on_board, colour, env.board, action)

    def position_state_from_env(self, env, player_color=None):
        colour = _normalise_colour(player_color or env.get_current_player())
        own_positions, opp_positions = _position_state_from_pins(env.pins_on_board, colour)
        return _reference_position_state(own_positions, opp_positions, env.board, colour)

    def choose_action(self, env, valid_actions, explore=False, time_budget_seconds=None):
        if not valid_actions:
            return None

        current_player = _normalise_colour(env.get_current_player())
        return self.choose_action_from_board(
            env.pins_on_board,
            current_player,
            valid_actions,
            env.board,
            explore=explore,
            time_budget_seconds=time_budget_seconds,
            turn_count=getattr(env, "turn_count", None),
            move_history=getattr(env, "move_history", None),
        )

    def _board_progress_score(self, pins_on_board, player_color, board, action, pin_map, target_indices):
        pin_id, destination = action
        origin = pin_map.get(int(pin_id))
        if origin is None:
            return -1000.0

        positions = [
            int(pin.axialindex)
            for pin in pins_on_board
            if _normalise_colour(pin.color) == _normalise_colour(player_color)
        ]
        old_distance = _assignment_distance(board, positions, target_indices)
        moved_positions = [int(destination) if pos == origin else pos for pos in positions]
        new_distance = _assignment_distance(board, moved_positions, target_indices)

        score = float(old_distance - new_distance)
        old_in_target = origin in target_indices
        new_in_target = int(destination) in target_indices
        if new_in_target and not old_in_target:
            score += 20.0
        if old_in_target and not new_in_target:
            score -= 30.0
        return score

    def _board_straggler_progress_score(self, pins_on_board, player_color, board, action, pin_map, target_indices):
        pin_id, destination = action
        origin = pin_map.get(int(pin_id))
        if origin is None:
            return -1000.0

        positions = [
            int(pin.axialindex)
            for pin in pins_on_board
            if _normalise_colour(pin.color) == _normalise_colour(player_color)
        ]
        old_distance = _squared_assignment_distance(board, positions, target_indices)
        moved_positions = [int(destination) if pos == origin else pos for pos in positions]
        new_distance = _squared_assignment_distance(board, moved_positions, target_indices)

        score = float(old_distance - new_distance)
        if int(destination) in target_indices and origin not in target_indices:
            score += 10.0
        if origin in target_indices and int(destination) not in target_indices:
            score -= 15.0
        return score

    def _compute_distance_map(self, board, player_color):
        # Single step BFS from target cells Cached per colour because the
        # board geometry never changes during a game
        colour_key = str(player_color)
        cache = getattr(self, "_distance_map_cache", None)
        if cache is None:
            cache = {}
            self._distance_map_cache = cache
        if colour_key in cache:
            return cache[colour_key]

        target_colour = board.colour_opposites.get(colour_key, "")
        targets = set(board.axial_of_colour(target_colour))
        if not targets:
            cache[colour_key] = {}
            return cache[colour_key]

        dist = {}
        queue = deque()
        for target_idx in targets:
            dist[int(target_idx)] = 0
            queue.append(int(target_idx))

        while queue:
            current = queue.popleft()
            current_distance = dist[current]
            cell = board.cells[current]
            for dq, dr in DIRECTIONS:
                neighbour = board.hole_index_of.get((cell.q + dq, cell.r + dr))
                if neighbour is None or neighbour in dist:
                    continue
                dist[int(neighbour)] = current_distance + 1
                queue.append(int(neighbour))

        cache[colour_key] = dist
        return dist

    def _bfs_straggler_action(self, pin_map, target_indices, valid_actions, board, player_color):
        # Push the farthest remaining straggler toward goal in late cleanup
        dist_map = self._compute_distance_map(board, player_color)
        if not dist_map:
            return None

        positions = [int(position) for position in pin_map.values()]
        old_assignment_distance = _assignment_distance(board, positions, target_indices)
        outside_pins = [
            (int(pin_id), int(position))
            for pin_id, position in pin_map.items()
            if position not in target_indices
        ]
        if not outside_pins:
            return None

        outside_pins.sort(key=lambda kv: dist_map.get(kv[1], 0), reverse=True)

        for pin_id, position in outside_pins:
            current_dist = dist_map.get(position, 0)
            origin_cell = board.cells[position]
            candidates = []
            for action_pin_id, destination in valid_actions:
                if int(action_pin_id) != pin_id:
                    continue
                new_dist = dist_map.get(int(destination))
                if new_dist is None or new_dist >= current_dist:
                    continue
                dest_cell = board.cells[int(destination)]
                chebyshev = max(
                    abs(origin_cell.q - dest_cell.q),
                    abs(origin_cell.r - dest_cell.r),
                    abs((-origin_cell.q - origin_cell.r) - (-dest_cell.q - dest_cell.r)),
                )
                moved_positions = [
                    int(destination) if pos == position else pos
                    for pos in positions
                ]
                new_assignment_distance = _assignment_distance(
                    board,
                    moved_positions,
                    target_indices,
                )
                bfs_gain = current_dist - new_dist
                squared_bfs_gain = current_dist * current_dist - new_dist * new_dist
                assignment_gain = old_assignment_distance - new_assignment_distance
                candidates.append((
                    squared_bfs_gain,
                    assignment_gain,
                    bfs_gain,
                    1 if int(destination) in target_indices else 0,
                    1 if chebyshev > 1 else 0,
                    (pin_id, int(destination)),
                ))

            if candidates:
                candidates.sort(reverse=True)
                return candidates[0][5]

        return None

    def _exact_reverse_penalties(self, pin_map, valid_actions, player_color, move_history):
        if not move_history:
            return [0.0] * len(valid_actions)

        penalties = []
        recent = list(move_history)[-12:]
        for pin_id, destination in valid_actions:
            origin = pin_map.get(int(pin_id))
            penalty = 0.0
            if origin is not None:
                for colour, old_pin_id, old_cell, new_cell in recent:
                    if (
                        _normalise_colour(colour) == _normalise_colour(player_color)
                        and int(old_pin_id) == int(pin_id)
                        and int(old_cell) == int(destination)
                        and int(new_cell) == int(origin)
                    ):
                        penalty = 0.35
                        break
            penalties.append(penalty)
        return penalties

    def _positions_by_pin_id(self, pins_on_board, player_color):
        by_id = {
            int(pin.id): int(pin.axialindex)
            for pin in pins_on_board
            if _normalise_colour(pin.color) == _normalise_colour(player_color)
        }
        return [by_id[pin_id] for pin_id in sorted(by_id)]

    def _one_move_winning_destinations(self, pins_on_board, player_color, board):
        positions = self._positions_by_pin_id(pins_on_board, player_color)
        target_indices = _target_positions(board, player_color)
        if len(positions) != 10 or sum(pos in target_indices for pos in positions) < 9:
            return set()

        occupied = {int(pin.axialindex) for pin in pins_on_board}
        winning_destinations = set()
        for pin_id, origin in enumerate(positions):
            if origin in target_indices:
                continue
            for destination in _legal_destinations(origin, occupied, board, player_color):
                moved = list(positions)
                moved[pin_id] = int(destination)
                if sum(pos in target_indices for pos in moved) == 10:
                    winning_destinations.add(int(destination))
        return winning_destinations

    def _emergency_block_action(self, pins_on_board, player_color, valid_actions, board, pin_map, target_indices, value_scores):
        opponent = board.colour_opposites.get(_normalise_colour(player_color), "")
        if not opponent:
            return None

        winning_destinations = self._one_move_winning_destinations(pins_on_board, opponent, board)
        if len(winning_destinations) != 1:
            return None

        block_cell = next(iter(winning_destinations))
        candidates = []
        for index, (pin_id, destination) in enumerate(valid_actions):
            if int(destination) != block_cell:
                continue
            origin = pin_map.get(int(pin_id))
            candidates.append((
                0 if origin in target_indices else 1,
                self._board_progress_score(
                    pins_on_board,
                    player_color,
                    board,
                    (pin_id, destination),
                    pin_map,
                    target_indices,
                ),
                value_scores[index],
                (int(pin_id), int(destination)),
            ))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][3]

    def _target_blocker_escape_count(self, blockers, target_indices):
        return sum(
            1
            for blocker in blockers
            for destination in blocker.get_legal_moves()
            if int(destination) not in target_indices
        )

    def _target_blocker_escape_action(self, pins_on_board, player_color, valid_actions, board, pin_map, target_indices, pins_in_goal, value_scores):
        # Random like opponents can strand a piece in their own home triangle,
        # which is the active target If that blocker has no legal way out, filling
        # around it can turn an otherwise won game into a 500 turn draw This
        # rescue is deliberately narrow late cleanup, confirmed target blockers,
        # and candidate moves that create escape moves for the blocker
        if pins_in_goal < 7:
            return None

        blockers = [
            pin for pin in pins_on_board
            if _normalise_colour(pin.color) != _normalise_colour(player_color)
            and int(pin.axialindex) in target_indices
        ]
        if not blockers:
            return None

        current_exits = self._target_blocker_escape_count(blockers, target_indices)
        if current_exits > 0:
            return None

        own_pins = {
            int(pin.id): pin
            for pin in pins_on_board
            if _normalise_colour(pin.color) == _normalise_colour(player_color)
        }

        candidates = []
        for index, (pin_id, destination) in enumerate(valid_actions):
            pin = own_pins.get(int(pin_id))
            if pin is None:
                continue

            origin = int(pin.axialindex)
            destination = int(destination)
            board.cells[origin].occupied = False
            pin.axialindex = destination
            board.cells[destination].occupied = True
            try:
                new_exits = self._target_blocker_escape_count(blockers, target_indices)
            finally:
                board.cells[destination].occupied = False
                pin.axialindex = origin
                board.cells[origin].occupied = True

            if new_exits <= current_exits:
                continue

            own_after = pins_in_goal
            if origin in target_indices and destination not in target_indices:
                own_after -= 1
            elif origin not in target_indices and destination in target_indices:
                own_after += 1

            candidates.append((
                new_exits - current_exits,
                new_exits,
                own_after,
                self._board_progress_score(
                    pins_on_board,
                    player_color,
                    board,
                    (int(pin_id), destination),
                    pin_map,
                    target_indices,
                ),
                value_scores[index],
                (int(pin_id), destination),
            ))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][5]

    def choose_action_from_board(self, pins_on_board, player_color, valid_actions, board, explore=False, time_budget_seconds=None, turn_count=None, move_history=None, player_order=None):
        if not valid_actions:
            return None

        if random.random() <= self.epsilon:
            return random.choice(valid_actions)

        features = [
            encode_afterstate(pins_on_board, player_color, board, action)
            for action in valid_actions
        ]
        feature_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            values = self.value_network(feature_tensor).squeeze(1)

        if explore:
            scored_actions = [
                (float(values[index].item()), action)
                for index, action in enumerate(valid_actions)
            ]
            return _choose_from_top_actions(scored_actions)

        value_scores = [float(value.item()) for value in values]
        normalized_player = _normalise_colour(player_color)
        target_indices = _target_positions(board, normalized_player)
        pin_map = {
            int(pin.id): int(pin.axialindex)
            for pin in pins_on_board
            if _normalise_colour(pin.color) == normalized_player
        }
        pins_in_goal = sum(1 for idx in pin_map.values() if idx in target_indices)
        distance = _assignment_distance(board, list(pin_map.values()), target_indices)
        reverse_penalties = self._exact_reverse_penalties(
            pin_map,
            valid_actions,
            normalized_player,
            move_history,
        )
        selection_scores = [
            value_scores[index] - reverse_penalties[index]
            for index in range(len(value_scores))
        ]
        best_index = max(range(len(valid_actions)), key=lambda i: selection_scores[i])

        if pins_in_goal >= 9:
            finish_indices = [
                index for index, (pin_id, destination) in enumerate(valid_actions)
                if int(destination) in target_indices
                and pin_map.get(int(pin_id)) not in target_indices
            ]
            if finish_indices:
                return valid_actions[max(finish_indices, key=lambda i: selection_scores[i])]

        emergency_block = self._emergency_block_action(
            pins_on_board,
            normalized_player,
            valid_actions,
            board,
            pin_map,
            target_indices,
            selection_scores,
        )
        if emergency_block is not None:
            return emergency_block

        target_unblock = self._target_blocker_escape_action(
            pins_on_board,
            normalized_player,
            valid_actions,
            board,
            pin_map,
            target_indices,
            pins_in_goal,
            selection_scores,
        )
        if target_unblock is not None:
            return target_unblock

        if pins_in_goal >= 9:
            # At 9/10 pins, finishing the last straggler dominates small value differences
            if turn_count is not None and int(turn_count) >= 80:
                progress_scores = [
                    self._board_progress_score(
                        pins_on_board,
                        normalized_player,
                        board,
                        action,
                        pin_map,
                        target_indices,
                    )
                    for action in valid_actions
                ]
                straggler_scores = [
                    self._board_straggler_progress_score(
                        pins_on_board,
                        normalized_player,
                        board,
                        action,
                        pin_map,
                        target_indices,
                    )
                    for action in valid_actions
                ]
                straggler_indices = [
                    index for index, (pin_id, _) in enumerate(valid_actions)
                    if pin_map.get(int(pin_id)) not in target_indices
                ]
                candidate_indices = straggler_indices or list(range(len(valid_actions)))
                progress_index = max(
                    candidate_indices,
                    key=lambda i: (progress_scores[i], straggler_scores[i], selection_scores[i]),
                )
                if (
                    progress_scores[progress_index] > 0.0
                    and selection_scores[progress_index] >= selection_scores[best_index] - 1.25
                ):
                    return valid_actions[progress_index]

        # If only a few pins remain outside target, prioritize the farthest one
        outside_count = len(pin_map) - pins_in_goal
        if 1 <= outside_count <= 3:
            straggler_action = self._bfs_straggler_action(
                pin_map,
                target_indices,
                valid_actions,
                board,
                normalized_player,
            )
            if straggler_action is not None:
                return straggler_action

        if turn_count is not None and int(turn_count) >= 200:
            progress_scores = [
                self._board_progress_score(
                    pins_on_board,
                    normalized_player,
                    board,
                    action,
                    pin_map,
                    target_indices,
                )
                for action in valid_actions
            ]
            straggler_scores = [
                self._board_straggler_progress_score(
                    pins_on_board,
                    normalized_player,
                    board,
                    action,
                    pin_map,
                    target_indices,
                )
                for action in valid_actions
            ]
            progress_index = max(
                range(len(valid_actions)),
                key=lambda i: (progress_scores[i], straggler_scores[i], selection_scores[i]),
            )
            # Very late low distance positions need progress even if value is flat
            if (
                distance <= 12
                and progress_scores[best_index] <= 0.0
                and progress_scores[progress_index] >= progress_scores[best_index] + 2.0
                and selection_scores[progress_index] >= selection_scores[best_index] - 1.0
            ):
                return valid_actions[progress_index]

        # Multiplayer override shared board midgames use progress scoring
        # unless an endgame heuristic above has already selected a move
        unique_colours = {_normalise_colour(str(pin.color)) for pin in pins_on_board}
        if len(unique_colours) > 2:
            # Dense multiplayer races need leader aware pressure long jumps
            # stay a tiebreaker rather than overpowering direct progress
            dense_multi_player = len(unique_colours) >= 5
            behind_leader = False
            if dense_multi_player:
                leader_distance = float("inf")
                for opp_colour in unique_colours:
                    if opp_colour == normalized_player:
                        continue
                    opp_target = _target_positions(board, opp_colour)
                    opp_positions = [
                        int(pin.axialindex)
                        for pin in pins_on_board
                        if _normalise_colour(str(pin.color)) == opp_colour
                    ]
                    if not opp_positions or not opp_target:
                        continue
                    opp_dist = _assignment_distance(board, opp_positions, opp_target)
                    if opp_dist < leader_distance:
                        leader_distance = opp_dist

                my_distance_now = _assignment_distance(board, list(pin_map.values()), target_indices)
                behind_leader = (
                    leader_distance != float("inf")
                    and (my_distance_now - leader_distance) > self.dense_multiplayer_leader_gap
                )

            mp_scored = []
            for action in valid_actions:
                base = self._board_progress_score(
                    pins_on_board,
                    normalized_player,
                    board,
                    action,
                    pin_map,
                    target_indices,
                )
                straggler = self._board_straggler_progress_score(
                    pins_on_board,
                    normalized_player,
                    board,
                    action,
                    pin_map,
                    target_indices,
                )
                jump_bonus = 0
                if behind_leader:
                    action_pin_id, action_destination = action
                    action_origin = pin_map.get(int(action_pin_id))
                    if action_origin is not None:
                        origin_cell = board.cells[int(action_origin)]
                        dest_cell = board.cells[int(action_destination)]
                        chebyshev = max(
                            abs(origin_cell.q - dest_cell.q),
                            abs(origin_cell.r - dest_cell.r),
                            abs((-origin_cell.q - origin_cell.r) - (-dest_cell.q - dest_cell.r)),
                        )
                        jump_bonus = max(0, chebyshev - 1)
                mp_scored.append((base, straggler, jump_bonus))

            mp_index = max(
                range(len(valid_actions)),
                key=lambda i: (
                    mp_scored[i][0],
                    mp_scored[i][1],
                    self.dense_multiplayer_jump_weight * mp_scored[i][2],
                    mp_scored[i][2],
                    selection_scores[i],
                ),
            )
            return valid_actions[mp_index]

        # Speed tie breaker for near equal value scores that otherwise stall progress
        if len(valid_actions) >= 2:
            best_score = selection_scores[best_index]
            tied_indices = [
                index for index, score in enumerate(selection_scores)
                if score >= best_score - 0.5
            ]
            if len(tied_indices) >= 2:
                best_progress = self._board_progress_score(
                    pins_on_board,
                    normalized_player,
                    board,
                    valid_actions[best_index],
                    pin_map,
                    target_indices,
                )
                if best_progress <= 0.0:
                    best_alt_index = None
                    best_alt_progress = best_progress
                    for index in tied_indices:
                        if index == best_index:
                            continue
                        alt_progress = self._board_progress_score(
                            pins_on_board,
                            normalized_player,
                            board,
                            valid_actions[index],
                            pin_map,
                            target_indices,
                        )
                        if alt_progress > best_alt_progress + 1.0:
                            best_alt_index = index
                            best_alt_progress = alt_progress
                    if best_alt_index is not None:
                        return valid_actions[best_alt_index]

        return valid_actions[best_index]

    def _append_experience(self, afterstate, reward, next_position_state, done, steps, demo=False):
        # Stored experiences stay lightweight so longer runs fit comfortably in RAM
        if next_position_state is not None:
            own_positions, opp_positions = next_position_state
            stored_next = (
                tuple(int(pos) for pos in own_positions),
                tuple(int(pos) for pos in opp_positions),
            )
        else:
            stored_next = None

        experience = (
            list(afterstate),
            float(reward),
            stored_next,
            bool(done),
            int(max(1, steps)),
        )
        if demo:
            self.demo_memory.append(experience)
        else:
            self.memory.append(experience)

    def _commit_pending(self, force=False):
        if not self.pending_experiences:
            return False
        if not force and len(self.pending_experiences) < self.n_step:
            return False

        horizon = min(self.n_step, len(self.pending_experiences))
        origin = self.pending_experiences[0]
        reward_total = 0.0
        bootstrap_state = None
        terminal = False
        steps = 0

        for sample in list(self.pending_experiences)[:horizon]:
            reward_total += (self.gamma ** steps) * float(sample["reward"])
            bootstrap_state = sample["next"]
            terminal = bool(sample["done"])
            steps += 1
            if terminal:
                bootstrap_state = None
                break

        self._append_experience(
            origin["afterstate"],
            reward_total,
            bootstrap_state,
            terminal,
            steps=steps,
            demo=origin["demo"],
        )
        self.pending_experiences.popleft()
        return True

    def remember(self, afterstate, reward, next_position_state, done, demo=False):
        # Short multi step targets usually give cleaner credit assignment than
        # a pure one step bootstrap, especially in longer Chinese Checkers races
        self.pending_experiences.append({
            "afterstate": list(afterstate),
            "reward": float(reward),
            "next": next_position_state,
            "done": bool(done),
            "demo": bool(demo),
        })

        self._commit_pending(force=False)

        if done:
            self.finish_episode()

    def finish_episode(self):
        # Some episodes end because the outer loop breaks instead of because a
        # stored transition was terminal Flushing here keeps the tail
        while self._commit_pending(force=True):
            pass

    def _sample_batch(self):
        # Mix supervised and online replay so updates do not erase guided examples
        combined_size = len(self.memory) + len(self.demo_memory)
        if combined_size < self.batch_size:
            return None

        demo_target = min(len(self.demo_memory), int(self.batch_size * self.demo_fraction))
        online_target = min(len(self.memory), self.batch_size - demo_target)

        if demo_target + online_target < self.batch_size:
            combined = list(self.demo_memory) + list(self.memory)
            return random.sample(combined, self.batch_size)

        batch = []
        if online_target:
            batch.extend(random.sample(self.memory, online_target))
        if demo_target:
            batch.extend(random.sample(self.demo_memory, demo_target))
        random.shuffle(batch)
        return batch

    def replay(self):
        batch = self._sample_batch()
        if batch is None:
            return False

        states = torch.FloatTensor([sample[0] for sample in batch]).to(self.device)
        current_values = self.value_network(states).squeeze(1)

        rewards_dones      = []
        groups             = []
        flat_afterstates   = []
        group_offsets      = []

        for sample in batch:
            if len(sample) == 5:
                _, reward, next_pos, done, steps_ahead = sample
            else:
                _, reward, next_pos, done = sample
                steps_ahead = 1
            rewards_dones.append((float(reward), bool(done), int(max(1, steps_ahead))))
            if not done and next_pos is not None:
                g = generate_afterstates_from_position_state(
                        next_pos, self.replay_board, self.player_color)
                groups.append(g)
                start = len(flat_afterstates)
                flat_afterstates.extend(g)
                group_offsets.append((start, len(flat_afterstates)))
            else:
                groups.append(None)
                group_offsets.append(None)

        # Batch generated next afterstates into one target network call
        flat_values = None
        if flat_afterstates:
            with torch.no_grad():
                flat_t = torch.FloatTensor(flat_afterstates).to(self.device)
                flat_values = self.target_network(flat_t).squeeze(1)

        targets = []
        for i, (reward, done, steps_ahead) in enumerate(rewards_dones):
            if group_offsets[i] is None or flat_values is None:
                targets.append(reward)
            else:
                s, e = group_offsets[i]
                if e > s:
                    targets.append(reward + (self.gamma ** steps_ahead) * flat_values[s:e].max().item())
                else:
                    targets.append(reward)

        target_tensor = torch.FloatTensor(targets).to(self.device)
        loss = nn.SmoothL1Loss()(current_values, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.train_updates += 1
        return True

    def save_model(self, filepath):
        torch.save({
            "value_network_state_dict": self.value_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_size": self.state_size,
            "memory_size": len(self.memory),
            "demo_memory_size": len(self.demo_memory),
            "train_updates": self.train_updates,
            "player_color": self.player_color,
            "board_perspective": "six_lane_rotation_v1",
            "n_step": self.n_step,
        }, filepath)
    def load_model(self, filepath, verbose=False):
        checkpoint = torch.load(filepath, map_location="cpu")
        if checkpoint.get("state_size") != self.state_size:
            print("Warning: saved afterstate model shape does not match current code.")
            return False

        try:
            self.value_network.load_state_dict(checkpoint["value_network_state_dict"])
            self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.train_updates = checkpoint.get("train_updates", 0)
            # Replay regenerates afterstates in the shared reference frame, so
            # runtime colour stays fixed even for lane specific older checkpoints
            self.player_color = _REFERENCE_COLOR
            demo_size = checkpoint.get("demo_memory_size", 0)
            if verbose:
                print(f"Model loaded from {filepath}")
                print(
                    f"Saved epsilon: {self.epsilon:.3f}  memory: {checkpoint['memory_size']}  "
                    f"demo: {demo_size}"
                )
            return True
        except RuntimeError as exc:
            print(f"Warning: could not load afterstate weights ({exc}).")
            return False

class AfterstateSearchAgent(AfterstateValueAgent):

    def __init__(self, state_size, player_color="yellow", name="AfterstateSearchAgent"):
        super().__init__(state_size=state_size, player_color=player_color, name=name)
        self.search_depth = 2
        self.endgame_depth = 3
        self.search_width = 10
        self.endgame_width = 12
        self.response_width = 6
        self.endgame_response_width = 8
        self.reply_weight = 0.85
        self.response_weight = self.reply_weight
        self.width_margin = 0.12
        self.cruise_margin = 28.0
        self.cruise_goal_pins = 6
        self.panic_margin = 10.0
        self.panic_goal_pins = 8
        self.panic_distance = 14
        self.cruise_gap = 0.35
        self.completion_weight = 0.65
        self.repeat_penalty = 8.0
        self.finisher_goal_pins = 6
        self.finisher_distance = 34
        self.finisher_margin = 24.0
        self.finisher_late_turn = 90
        self.default_time_budget_seconds = 1.25
        self.search_turn_mode = "duel"
        self.search_opponent_mode = "first"
        self._value_cache = {}
        self._search_cache = {}
        self._search_deadline = None

    def _time_exceeded(self):
        return self._search_deadline is not None and time.time() >= self._search_deadline

    def _repeat_count(self, state):
        repetition_key = state.repetition_key()
        return int(state.state_counts.get(repetition_key, 0))

    def _encode_state_action(self, state, player_color, action):
        colour = _normalise_colour(player_color)
        opponent = state.opponent_of(colour)
        own_positions = tuple(state.positions.get(colour, ()))
        opp_positions = tuple(state.positions.get(opponent, ()))

        pin_id, destination = action
        moved_positions = list(own_positions)
        if 0 <= int(pin_id) < len(moved_positions):
            moved_positions[int(pin_id)] = int(destination)
        moved_positions.sort()
        return _encode_from_positions(tuple(moved_positions), opp_positions, state.board, colour)

    def _state_signature(self, state):
        # Search caches should distinguish both the board layout and how close
        # that layout is to a repetition draw or max turn draw
        repetition_key = state.repetition_key()
        repeat_count = int(state.state_counts.get(repetition_key, 0))
        return repetition_key, int(state.move_count), repeat_count

    def _afterstate_value(self, state, player_color, action):
        cache_key = (self._state_signature(state), str(player_color), int(action[0]), int(action[1]))
        cached = self._value_cache.get(cache_key)
        if cached is not None:
            return cached

        feature = self._encode_state_action(state, player_color, action)
        feature_tensor = torch.FloatTensor(feature).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.value_network(feature_tensor).squeeze(1)
        scored = float(value.item())
        self._value_cache[cache_key] = scored
        return scored

    def _search_budget(self, state, root_colour):
        # Search gets more serious once either side is close to finishing, while
        # staying cheaper in the opening when most moves are setup moves
        root = _normalise_colour(root_colour)
        opponent = state.opponent_of(root)
        root_pins = state.pins_in_goal(root)
        opponent_pins = state.pins_in_goal(opponent)
        root_distance = state.total_distance_to_target(root)
        opponent_distance = state.total_distance_to_target(opponent)
        race_margin = state.race_margin(root)

        # With one pin left, prioritize completion even when the opponent is not close
        if root_pins >= 9:
            return (
                self.endgame_depth + 1,
                self.endgame_width + 2,
                self.endgame_response_width + 2,
            )

        # Spend extra search in tight races or defensive emergencies
        if (
            abs(race_margin) <= self.panic_margin
            or opponent_pins >= self.panic_goal_pins
            or opponent_distance <= self.panic_distance
        ):
            return (
                self.endgame_depth + 1,
                self.endgame_width + 2,
                self.endgame_response_width + 2,
            )

        # Reduce depth when the position is already comfortably ahead
        if (
            race_margin >= self.cruise_margin
            and root_pins >= self.cruise_goal_pins
            and root_distance + 8 < opponent_distance
        ):
            return (
                max(1, self.search_depth - 1),
                max(4, self.search_width - 2),
                max(3, self.response_width - 1),
            )

        if (
            root_pins >= 7 or opponent_pins >= 7
            or root_distance <= 18 or opponent_distance <= 18
        ):
            return self.endgame_depth, self.endgame_width, self.endgame_response_width

        if (
            root_pins >= 4 or opponent_pins >= 4
            or root_distance <= 34 or opponent_distance <= 34
        ):
            return self.search_depth, max(self.search_width, 12), max(self.response_width, 7)

        return self.search_depth, self.search_width, self.response_width

    def _cruise_action(self, state, root_colour, valid_actions):
        # In a comfortable lead, prefer clean finishing progress over deeper reply search
        root = _normalise_colour(root_colour)
        ranked = []
        for action in valid_actions:
            ranked.append(
                (
                    state.endgame_progress_score(action, root),
                    self._afterstate_value(state, root, action),
                    action,
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return ranked

    def _should_use_finisher(self, state, root_colour):
        root = _normalise_colour(root_colour)
        opponent = state.opponent_of(root)
        root_pins = state.pins_in_goal(root)
        root_distance = state.total_distance_to_target(root)
        opponent_pins = state.pins_in_goal(opponent)
        opponent_distance = state.total_distance_to_target(opponent)
        race_margin = state.race_margin(root)

        clearly_ahead = (
            race_margin >= self.finisher_margin
            and opponent_distance >= root_distance + 12
            and opponent_pins <= max(6, root_pins)
        )
        nearly_done = root_pins >= self.finisher_goal_pins or root_distance <= self.finisher_distance
        late_and_ahead = state.move_count >= self.finisher_late_turn and race_margin >= self.finisher_margin
        return clearly_ahead and (nearly_done or late_and_ahead)

    def _finisher_action(self, state, root_colour, valid_actions):
        root = _normalise_colour(root_colour)
        root_pins = state.pins_in_goal(root)
        root_distance = state.total_distance_to_target(root)
        root_straggler = state.straggler_penalty(root)
        target_positions = state.target_cells(root)
        positions = state.positions.get(root, ())

        ranked = []
        for action in valid_actions:
            pin_id, destination = action
            next_state, done, info = state.apply_action(action)
            if done and info.get("winner") == root:
                return action

            next_pins = next_state.pins_in_goal(root)
            next_distance = next_state.total_distance_to_target(root)
            next_straggler = next_state.straggler_penalty(root)

            pins_gain = next_pins - root_pins
            distance_gain = root_distance - next_distance
            straggler_gain = root_straggler - next_straggler
            old_position = int(positions[int(pin_id)]) if 0 <= int(pin_id) < len(positions) else -1
            old_in_target = old_position in target_positions
            new_in_target = int(destination) in target_positions

            score = 900.0 * pins_gain
            score += 60.0 * distance_gain
            score += 18.0 * straggler_gain
            score += 4.0 * state.endgame_progress_score(action, root)
            if old_in_target and not new_in_target:
                score -= 600.0
            elif old_in_target and next_pins < 10:
                score -= 60.0
            if new_in_target and not old_in_target:
                score += 120.0

            repeat_count = int(next_state.state_counts.get(next_state.repetition_key(), 0))
            if repeat_count > 1:
                score -= 250.0 * float(repeat_count - 1)

            ranked.append((score, self._afterstate_value(state, root, action), action))

        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return ranked[0][2]

    def _adaptive_width(self, scored_actions, base_width, max_width):
        # If several actions look almost tied, widening the frontier a little
        # is usually cheaper than paying for a full deep search on every move
        if not scored_actions:
            return 0

        width = min(len(scored_actions), int(base_width))
        limit = min(len(scored_actions), int(max_width))
        boundary = scored_actions[width - 1][0]

        while width < limit:
            if abs(scored_actions[width][0] - boundary) > self.width_margin:
                break
            width += 1
            boundary = scored_actions[width - 1][0]

        return width

    def _terminal_score(self, state, root_colour):
        root = _normalise_colour(root_colour)
        opponent = state.opponent_of(root)
        if state.pins_in_goal(root) == 10:
            return 1000.0
        if state.pins_in_goal(opponent) == 10:
            return -1000.0
        return None

    def _completion_score(self, state, root_colour):
        # Search should prefer finishing over endlessly polishing a strong looking
        # position These terms get sharper late
        root = _normalise_colour(root_colour)
        opponent = state.opponent_of(root)

        root_pins = state.pins_in_goal(root)
        opponent_pins = state.pins_in_goal(opponent)
        root_distance = state.total_distance_to_target(root)
        opponent_distance = state.total_distance_to_target(opponent)
        root_straggler = float(state.straggler_penalty(root))
        opponent_straggler = float(state.straggler_penalty(opponent))
        root_unfinished = max(0, 10 - root_pins)
        opponent_unfinished = max(0, 10 - opponent_pins)

        score = 18.0 * (root_pins - opponent_pins)
        score += 0.60 * (opponent_distance - root_distance)
        score += 2.5 * (opponent_straggler - root_straggler)

        # Once only a few pins remain, finishing those stragglers should
        # dominate subtle positional improvements
        if root_unfinished <= 3:
            score -= 2.0 * root_distance
            score -= 5.0 * root_straggler
        elif root_unfinished <= 5:
            score -= 1.0 * root_distance
            score -= 2.5 * root_straggler

        # Late games penalize unfinished work unless the move reduces it
        late_factor = max(0.0, float(state.move_count) - 140.0) / 40.0
        if late_factor > 0.0:
            score -= late_factor * (
                1.6 * root_unfinished
                + 0.20 * root_distance
                + 1.2 * root_straggler
            )
            score += 0.35 * late_factor * (
                opponent_unfinished
                + 0.10 * opponent_distance
            )

        # Repeated states are penalized to avoid draw prone loops
        repeat_count = self._repeat_count(state)
        if repeat_count > 1:
            score -= self.repeat_penalty * float(repeat_count - 1)

        return score

    def _completion_delta(self, previous_state, next_state, root_colour):
        return self._completion_score(next_state, root_colour) - self._completion_score(previous_state, root_colour)

    def _draw_score(self, state, root_colour):
        # Drawn races are still better for the side that is closer to finishing,
        # but unfinished late draws should carry a meaningful penalty
        heuristic = 12.0 * state.heuristic_value(root_colour)
        return heuristic + self.completion_weight * self._completion_score(state, root_colour)

    def _terminal_score_from_info(self, state, root_colour, info):
        winner = info.get("winner")
        if winner is None:
            return self._draw_score(state, root_colour)
        if str(winner) == str(root_colour):
            return 1000.0
        return -1000.0

    def _leaf_score(self, state, root_colour):
        root = _normalise_colour(root_colour)
        current = _normalise_colour(state.current_player)
        heuristic = 12.0 * state.heuristic_value(root)
        completion = self.completion_weight * self._completion_score(state, root)

        if current == root:
            actions = state.valid_actions()
            if not actions:
                return heuristic + completion
            scored = [self._afterstate_value(state, root, action) for action in actions]
            return max(scored) + heuristic + completion

        return heuristic + completion

    def _ordered_actions(self, state, player_color, actions, maximizing):
        colour = _normalise_colour(player_color)
        _, search_width, response_width = self._search_budget(state, colour)
        if maximizing:
            # On the root players turn, search high value afterstates first
            scored = [
                (
                    self._afterstate_value(state, colour, action),
                    state.endgame_progress_score(action, colour),
                    action,
                )
                for action in actions
            ]
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            width = self._adaptive_width(scored, search_width, self.endgame_width)
            return [action for _, _, action in scored[:width]]

        # Opponent replies use the cheaper progress score to keep search light
        scored = [
            (
                state.progress_score(action, colour),
                action,
            )
            for action in actions
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        width = self._adaptive_width(scored, response_width, self.endgame_response_width)
        return [action for _, action in scored[:width]]

    def _search(self, state, root_colour, depth, alpha=None, beta=None):
        if self._time_exceeded():
            return self._leaf_score(state, root_colour)
        if alpha is None:
            alpha = float("-inf")
        if beta is None:
            beta = float("inf")
        cache_key = (self._state_signature(state), str(root_colour), int(depth))
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return cached

        terminal = self._terminal_score(state, root_colour)
        if terminal is not None:
            self._search_cache[cache_key] = terminal
            return terminal
        if depth <= 0:
            leaf = self._leaf_score(state, root_colour)
            self._search_cache[cache_key] = leaf
            return leaf

        current = _normalise_colour(state.current_player)
        root = _normalise_colour(root_colour)
        maximizing = current == root
        actions = state.valid_actions()
        if not actions:
            leaf = self._leaf_score(state, root_colour)
            self._search_cache[cache_key] = leaf
            return leaf

        ordered_actions = self._ordered_actions(state, current, actions, maximizing=maximizing)

        if maximizing:
            best_score = float("-inf")
            for action in ordered_actions:
                if self._time_exceeded():
                    break
                immediate = self._afterstate_value(state, current, action)
                next_state, done, info = state.apply_action(action)
                completion_gain = self._completion_delta(state, next_state, root_colour)
                if done:
                    reply_score = self._terminal_score_from_info(next_state, root_colour, info)
                else:
                    reply_score = self._search(next_state, root_colour, depth - 1, alpha, beta)
                score = immediate + 0.30 * completion_gain + self.reply_weight * reply_score
                if score > best_score:
                    best_score = score
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            self._search_cache[cache_key] = best_score
            return best_score

        worst_score = float("inf")
        for action in ordered_actions:
            if self._time_exceeded():
                break
            next_state, done, info = state.apply_action(action)
            if done:
                score = self._terminal_score_from_info(next_state, root_colour, info)
            else:
                score = self._search(next_state, root_colour, depth - 1, alpha, beta)
            if score < worst_score:
                worst_score = score
            beta = min(beta, worst_score)
            if beta <= alpha:
                break
        self._search_cache[cache_key] = worst_score
        return worst_score

    def choose_action(self, env, valid_actions, explore=False, time_budget_seconds=None):
        if not valid_actions:
            return None
        if random.random() <= self.epsilon:
            return random.choice(valid_actions)

        state = SearchBoardState.from_env(env)
        state.turn_mode = self.search_turn_mode
        state.opponent_mode = self.search_opponent_mode
        return self._choose_action_from_state(
            state,
            _normalise_colour(env.get_current_player()),
            valid_actions,
            explore=explore,
            time_budget_seconds=time_budget_seconds,
        )

    def _choose_action_from_state(self, state, root_colour, valid_actions, explore=False, time_budget_seconds=None):
        # Action selection starts with cheap safeguards, then spends time on reply search
        self._value_cache.clear()
        self._search_cache.clear()
        if time_budget_seconds is None:
            time_budget_seconds = self.default_time_budget_seconds
        self._search_deadline = (
            time.time() + max(0.01, float(time_budget_seconds))
            if time_budget_seconds is not None
            else None
        )

        try:
            root = _normalise_colour(root_colour)
            opponent = state.opponent_of(root)
            race_margin = state.race_margin(root)
            root_pins = state.pins_in_goal(root)
            root_distance = state.total_distance_to_target(root)
            opponent_distance = state.total_distance_to_target(opponent)

            # With one pin left, use the finisher heuristic to avoid terminal stalls
            # Keep the threshold at nine pins so normal search handles earlier endgames
            if not explore and root_pins >= 9:
                finisher_action = self._finisher_action(state, root, valid_actions)
                if finisher_action is not None:
                    return finisher_action

            if not explore and self._should_use_finisher(state, root):
                finisher_action = self._finisher_action(state, root, valid_actions)
                if finisher_action is not None:
                    return finisher_action

            ordered_actions = self._ordered_actions(state, root, valid_actions, maximizing=True)
            if not ordered_actions:
                return valid_actions[0]

            fallback_action = ordered_actions[0]

            if (
                not explore
                and race_margin >= self.cruise_margin
                and root_pins >= self.cruise_goal_pins
                and root_distance + 8 < opponent_distance
            ):
                cruise_ranked = self._cruise_action(state, root, valid_actions)
                if cruise_ranked:
                    if len(cruise_ranked) == 1:
                        return cruise_ranked[0][2]
                    best_progress, best_value, best_action = cruise_ranked[0]
                    next_progress, next_value, _ = cruise_ranked[1]
                    if (
                        best_progress > next_progress
                        or best_value >= next_value + self.cruise_gap
                    ):
                        return best_action

            if self._time_exceeded():
                return fallback_action

            best_action = fallback_action
            best_score = float("-inf")
            scored_actions = []
            depth_budget, _, _ = self._search_budget(state, root)

            # Score the candidate move, then subtract the best reply estimate
            for action in ordered_actions:
                if self._time_exceeded():
                    break
                immediate = self._afterstate_value(state, root, action)
                next_state, done, info = state.apply_action(action)
                completion_gain = self._completion_delta(state, next_state, root)
                if done:
                    reply_score = self._terminal_score_from_info(next_state, root, info)
                else:
                    reply_score = self._search(next_state, root, depth_budget - 1, float("-inf"), float("inf"))
                score = immediate + 0.30 * completion_gain + self.reply_weight * reply_score
                scored_actions.append((score, action))
                if score > best_score:
                    best_score = score
                    best_action = action

            if explore:
                exploratory_action = _choose_from_top_actions(scored_actions)
                if exploratory_action is not None:
                    return exploratory_action

            return best_action if best_action is not None else fallback_action
        finally:
            self._search_deadline = None

    def choose_action_from_board(self, pins_on_board, player_color, valid_actions, board, explore=False, time_budget_seconds=None, turn_count=None, move_history=None, player_order=None):
        if not valid_actions:
            return None

        if random.random() <= self.epsilon:
            return random.choice(valid_actions)

        root_colour = _normalise_colour(player_color)
        state = SearchBoardState.from_board(
            pins_on_board,
            root_colour,
            board,
            player_order=player_order,
            move_count=0 if turn_count is None else int(turn_count),
            turn_mode=self.search_turn_mode,
            opponent_mode=self.search_opponent_mode,
        )
        return self._choose_action_from_state(
            state,
            root_colour,
            valid_actions,
            explore=explore,
            time_budget_seconds=time_budget_seconds,
        )

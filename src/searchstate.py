# Immutable board snapshot used by the Afterstate search controller

import math

from src.board import HEX_DIRECTIONS
from src.perspective import (
    REFERENCE_COLOR,
    index_to_reference_perspective,
    indices_to_reference_perspective,
)

DIRECTIONS = HEX_DIRECTIONS
_REFERENCE_COLOR = REFERENCE_COLOR


def _colour_list(board, player_order=None):
    if player_order is not None:
        return [str(colour) for colour in player_order]

    colours = set()
    for colour, opposite in board.colour_opposites.items():
        colours.add(str(colour))
        colours.add(str(opposite))
    return sorted(colours)


def _infer_player_order_from_pins(pins_on_board, current_player):
    current = str(current_player)
    colours = [current]
    for pin in pins_on_board:
        colour = str(pin.color)
        if colour not in colours:
            colours.append(colour)
    return colours


def _positions_from_pins(pins_on_board, colours):
    # Positions are stored by pin id so action indices stay stable in search
    positions = {colour: [None] * 10 for colour in colours}

    for pin in pins_on_board:
        colour = str(pin.color)
        if colour not in positions:
            positions[colour] = [None] * 10
        pin_id = int(pin.id)
        if 0 <= pin_id < len(positions[colour]):
            positions[colour][pin_id] = int(pin.axialindex)

    for colour in positions:
        fallback = []
        for pin in pins_on_board:
            if str(pin.color) == colour:
                fallback.append((int(pin.id), int(pin.axialindex)))
        if fallback:
            fallback.sort()
            for pin_id, axialindex in fallback:
                if positions[colour][pin_id] is None:
                    positions[colour][pin_id] = axialindex
        positions[colour] = tuple(
            int(axialindex) if axialindex is not None else 0
            for axialindex in positions[colour]
        )

    return positions


def _axial_distance(board, idx1, idx2):
    cell_1 = board.cells[int(idx1)]
    cell_2 = board.cells[int(idx2)]
    q_diff = cell_1.q - cell_2.q
    r_diff = cell_1.r - cell_2.r
    s_diff = (-cell_1.q - cell_1.r) - (-cell_2.q - cell_2.r)
    return max(abs(q_diff), abs(r_diff), abs(s_diff))


def _count_pins_in_target(positions, target_cells):
    return sum(1 for position in positions if position in target_cells)


def _total_distance_to_target(board, positions, target_cells):
    if not positions or not target_cells:
        return 0

    in_target = {position for position in positions if position in target_cells}
    outside = [position for position in positions if position not in target_cells]
    if not outside:
        return 0

    available = [cell for cell in target_cells if cell not in in_target]
    if not available:
        return 0

    outside_sorted = sorted(
        outside,
        key=lambda position: min(_axial_distance(board, position, target) for target in available),
        reverse=True,
    )

    total = 0
    remaining = list(available)
    for position in outside_sorted:
        if not remaining:
            break
        best_target = min(remaining, key=lambda target: _axial_distance(board, position, target))
        total += _axial_distance(board, position, best_target)
        remaining.remove(best_target)

    return total


def _repetition_positions_key(positions_by_colour):
    # Repetition should care about occupied cells, not same colour pin ids
    return tuple(
        sorted(
            (str(colour), tuple(sorted(int(position) for position in positions)))
            for colour, positions in positions_by_colour.items()
        )
    )


class SearchBoardState:
    def __init__(
        self,
        board,
        positions,
        current_player,
        player_order=None,
        move_count=0,
        max_turns=400,
        max_repetitions=6,
        state_counts=None,
        turn_mode="cycle",
        opponent_mode="leader",
    ):
        self.board = board
        self.player_order = tuple(_colour_list(board, player_order))
        self.positions = {
            str(colour): tuple(int(position) for position in positions.get(str(colour), ()))
            for colour in self.player_order
        }
        self.current_player = str(current_player)
        self.move_count = int(move_count)
        self.max_turns = int(max_turns)
        self.max_repetitions = int(max_repetitions)
        self.state_counts = dict(state_counts or {})
        self.turn_mode = str(turn_mode)
        self.opponent_mode = str(opponent_mode)

    @classmethod
    def from_env(cls, env):
        colours = [str(colour) for colour in env.player_colors]
        positions = _positions_from_pins(env.pins_on_board, colours)
        return cls(
            board=env.board,
            positions=positions,
            current_player=env.get_current_player(),
            player_order=colours,
            move_count=env.turn_count,
            max_turns=env.max_turns,
            max_repetitions=env.max_repetitions,
            state_counts=env.state_counts,
        )

    @classmethod
    def from_board(
        cls,
        pins_on_board,
        current_player,
        board,
        player_order=None,
        move_count=0,
        max_turns=400,
        max_repetitions=6,
        state_counts=None,
        turn_mode="cycle",
        opponent_mode="leader",
    ):
        colours = _colour_list(
            board,
            player_order if player_order is not None else _infer_player_order_from_pins(pins_on_board, current_player),
        )
        positions = _positions_from_pins(pins_on_board, colours)
        return cls(
            board=board,
            positions=positions,
            current_player=current_player,
            player_order=colours,
            move_count=move_count,
            max_turns=max_turns,
            max_repetitions=max_repetitions,
            state_counts=state_counts,
            turn_mode=turn_mode,
            opponent_mode=opponent_mode,
        )

    def opponent_of(self, colour):
        colour = str(colour)
        opponents = [player for player in self.player_order if player != colour]
        if not opponents:
            return colour
        if len(opponents) == 1 or self.opponent_mode == "first":
            return opponents[0]
        return max(
            opponents,
            key=lambda player: (
                self.pins_in_goal(player),
                -self.total_distance_to_target(player),
            ),
        )

    def next_player_after(self, colour):
        colour = str(colour)
        if not self.player_order:
            return colour
        try:
            index = self.player_order.index(colour)
        except ValueError:
            return self.player_order[0]
        return self.player_order[(index + 1) % len(self.player_order)]

    def target_cells(self, colour):
        target_colour = self.board.colour_opposites.get(str(colour), "")
        return set(self.board.axial_of_colour(target_colour)) if target_colour else set()

    def pins_in_goal(self, colour):
        return _count_pins_in_target(self.positions.get(str(colour), ()), self.target_cells(colour))

    def total_distance_to_target(self, colour):
        return _total_distance_to_target(
            self.board,
            self.positions.get(str(colour), ()),
            self.target_cells(colour),
        )

    def straggler_penalty(self, colour):
        positions = self.positions.get(str(colour), ())
        target_cells = self.target_cells(colour)
        if not target_cells or len(positions) < 2:
            return 0.0
        outside = [pos for pos in positions if pos not in target_cells]
        if len(outside) < 2:
            return 0.0
        distances = [
            min(_axial_distance(self.board, pos, t) for t in target_cells)
            for pos in outside
        ]
        avg = sum(distances) / len(distances)
        return max(0.0, max(distances) - avg)

    def race_margin(self, player_color):
        colour = str(player_color)
        opponent = self.opponent_of(colour)
        my_pins = self.pins_in_goal(colour)
        opp_pins = self.pins_in_goal(opponent)
        my_dist = self.total_distance_to_target(colour)
        opp_dist = self.total_distance_to_target(opponent)
        return 18.0 * (my_pins - opp_pins) + 0.35 * (opp_dist - my_dist)

    def heuristic_value(self, player_color):
        margin = self.race_margin(player_color)
        return float(math.tanh(margin / 12.0))

    def state_vector(self, player_color=None):
        colour = str(player_color or self.current_player)
        own_positions = set(self.positions.get(colour, ()))
        opponent_positions = set()

        for other_colour, positions in self.positions.items():
            if other_colour != colour:
                opponent_positions.update(positions)

        if colour != _REFERENCE_COLOR:
            own_positions = set(indices_to_reference_perspective(self.board, colour, own_positions))
            opponent_positions = set(indices_to_reference_perspective(self.board, colour, opponent_positions))
            colour = _REFERENCE_COLOR

        target_positions = self.target_cells(colour)
        board_size = len(self.board.cells)
        own_layer = [1 if idx in own_positions else 0 for idx in range(board_size)]
        opp_layer = [1 if idx in opponent_positions else 0 for idx in range(board_size)]
        target_layer = [1 if idx in target_positions else 0 for idx in range(board_size)]
        return own_layer + opp_layer + target_layer

    def reference_destination(self, action, player_color=None):
        _, destination = action
        colour = str(player_color or self.current_player)
        return index_to_reference_perspective(self.board, colour, destination, _REFERENCE_COLOR)

    def policy_index(self, action, player_color=None):
        pin_id, _ = action
        return int(pin_id) * 121 + self.reference_destination(action, player_color)

    def repetition_key(self, current_player=None):
        return str(current_player or self.current_player), _repetition_positions_key(self.positions)

    def valid_actions(self, player_color=None):
        colour = str(player_color or self.current_player)
        positions = self.positions.get(colour, ())
        occupied = set()
        for colour_positions in self.positions.values():
            occupied.update(int(position) for position in colour_positions)

        valid = []
        for pin_id, start_idx in enumerate(positions):
            start_cell = self.board.cells[int(start_idx)]
            possible = set()

            for dq, dr in DIRECTIONS:
                neighbour = self.board.hole_index_of.get((start_cell.q + dq, start_cell.r + dr))
                if neighbour is not None and neighbour not in occupied:
                    possible.add(neighbour)

            visited = {int(start_idx)}
            stack = [int(start_idx)]
            while stack:
                current = stack.pop()
                current_cell = self.board.cells[int(current)]
                for dq, dr in DIRECTIONS:
                    adjacent = self.board.hole_index_of.get((current_cell.q + dq, current_cell.r + dr))
                    landing = self.board.hole_index_of.get((current_cell.q + 2 * dq, current_cell.r + 2 * dr))
                    if adjacent is None or landing is None:
                        continue
                    if adjacent not in occupied or landing in occupied or landing in visited:
                        continue
                    possible.add(landing)
                    visited.add(landing)
                    stack.append(landing)

            for destination in possible:
                if start_cell.postype != colour and self.board.cells[int(destination)].postype == colour:
                    continue
                valid.append((pin_id, int(destination)))

        return valid

    def progress_score(self, action, colour):
        pin_id, destination = action
        positions = list(self.positions.get(str(colour), ()))
        if not 0 <= int(pin_id) < len(positions):
            return -1.0

        old_position = positions[int(pin_id)]
        old_distance = _total_distance_to_target(self.board, positions, self.target_cells(colour))
        positions[int(pin_id)] = int(destination)
        new_distance = _total_distance_to_target(self.board, positions, self.target_cells(colour))

        score = float(old_distance - new_distance)
        target_positions = self.target_cells(colour)
        old_in_target = int(old_position) in target_positions
        new_in_target = int(destination) in target_positions
        if new_in_target and not old_in_target:
            score += 20.0
        if old_in_target and not new_in_target:
            score -= 80.0
        if old_in_target and new_in_target:
            score -= 1.0
        return score

    def endgame_progress_score(self, action, colour):
        score = self.progress_score(action, colour)
        pin_id, destination = action
        positions = self.positions.get(str(colour), ())
        if not 0 <= int(pin_id) < len(positions):
            return score

        target_positions = self.target_cells(colour)
        old_position = int(positions[int(pin_id)])
        old_in_target = old_position in target_positions
        new_in_target = int(destination) in target_positions
        unfinished_pins = sum(1 for position in positions if int(position) not in target_positions)

        if unfinished_pins > 1 and old_in_target:
            score -= 12.0
        elif unfinished_pins == 1 and old_in_target and new_in_target:
            score -= 2.0
        if not old_in_target:
            score += 4.0
            if new_in_target:
                score += 12.0
        return score

    def apply_action(self, action):
        pin_id, destination = action
        mover = self.current_player
        next_player = self.next_player_after(mover) if self.turn_mode == "cycle" else self.opponent_of(mover)
        new_positions = {colour: list(positions) for colour, positions in self.positions.items()}
        new_positions[mover][int(pin_id)] = int(destination)
        next_state = SearchBoardState(
            board=self.board,
            positions=new_positions,
            current_player=next_player,
            player_order=self.player_order,
            move_count=self.move_count + 1,
            max_turns=self.max_turns,
            max_repetitions=self.max_repetitions,
            turn_mode=self.turn_mode,
            opponent_mode=self.opponent_mode,
        )

        if next_state.pins_in_goal(mover) == len(next_state.positions.get(mover, ())):
            return next_state, True, {"winner": mover, "message": f"{mover} wins"}

        if next_state.move_count >= self.max_turns:
            return next_state, True, {"winner": None, "message": "draw"}

        next_counts = dict(self.state_counts)
        repetition_key = next_state.repetition_key()
        next_counts[repetition_key] = next_counts.get(repetition_key, 0) + 1
        next_state.state_counts = next_counts
        if next_state.max_repetitions > 0 and next_counts[repetition_key] >= next_state.max_repetitions:
            return next_state, True, {"winner": None, "message": "draw"}

        return next_state, False, {"winner": None, "message": ""}

    def terminal_value_for(self, player_color, info):
        winner = info.get("winner")
        if winner is None:
            return 0.0
        return 1.0 if str(winner) == str(player_color) else -1.0

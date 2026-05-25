# Checks for board rules and small helper functions

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.board import HexBoard
from src.env import ChineseCheckersEnv
from src.game import BASE_COLORS, OPPOSITE_COLORS, GameManager
from src.rewards import afterstate_shaped_reward, compute_potential, model_selection_score


def check_board_geometry():
    # The standard board has 121 playable cells
    board = HexBoard(R=4, hole_radius=16, spacing=34)
    assert len(board.cells) == 121

    # Each colour has exactly ten home cells
    for colour in BASE_COLORS + OPPOSITE_COLORS:
        assert len(board.axial_of_colour(colour)) == 10

    # Opposite colours must map back to each other
    for colour, opposite in board.colour_opposites.items():
        assert board.colour_opposites[opposite] == colour


def check_player_setup():
    np.random.seed(20260526)

    for player_count in (2, 3, 4, 5, 6):
        board = HexBoard(R=4, hole_radius=16, spacing=34)
        game = GameManager(board, num_players=player_count)
        colours = game.assign_players_colors(player_count)
        assert len(colours) == player_count
        assert len(set(colours)) == player_count

        pins = game.place_pins_to_board(colours)
        assert len(pins) == player_count * 10
        assert sum(1 for cell in board.cells if cell.occupied) == player_count * 10

        for colour in colours:
            assert sum(1 for pin in pins if pin.color == colour) == 10

    # Two player games must use an opposite colour lane
    board = HexBoard(R=4, hole_radius=16, spacing=34)
    game = GameManager(board, num_players=2)
    colours = game.assign_players_colors(2)
    assert board.colour_opposites[colours[0]] == colours[1]


def check_environment_step():
    env = ChineseCheckersEnv(num_players=2, player_colors=["red", "blue"], max_turns=10)
    state = env.reset()
    assert len(state) == 363
    assert env.get_current_player() == "red"
    assert env.get_valid_actions()

    # Invalid pin ids should not end the game
    next_state, reward, done, info = env.step((99, 0))
    assert len(next_state) == 363
    assert reward == -1
    assert not done
    assert info["message"] == "Invalid pin"

    # A real legal action should advance the turn counter
    action = env.get_valid_actions()[0]
    _, _, done, _ = env.step(action)
    assert env.turn_count == 1
    assert not done


def check_reward_helpers():
    env = ChineseCheckersEnv(num_players=2, player_colors=["red", "blue"])
    env.reset()
    assert isinstance(compute_potential(env, "red"), float)
    assert model_selection_score(50.0, 8.0) == 130.0

    win_reward = afterstate_shaped_reward(
        True,
        {"message": "red wins"},
        "red",
        phi_s=0.0,
        phi_sp=1.0,
        action_progress=1.0,
    )
    draw_reward = afterstate_shaped_reward(
        True,
        {"message": "draw"},
        "red",
        phi_s=0.0,
        phi_sp=1.0,
        action_progress=1.0,
    )
    loss_reward = afterstate_shaped_reward(
        True,
        {"message": "blue wins"},
        "red",
        phi_s=0.0,
        phi_sp=1.0,
        action_progress=1.0,
    )

    assert win_reward > draw_reward > loss_reward


def main():
    check_board_geometry()
    check_player_setup()
    check_environment_step()
    check_reward_helpers()
    print("Core logic checks passed.")


if __name__ == "__main__":
    main()

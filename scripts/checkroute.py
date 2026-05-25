# Checks for the tournament route

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import player
from scripts.comparemodels import parse_models
from scripts.trainmultiplayer import parse_mix
from scripts.trainmultiplayer import DEFAULT_BEST_OUTPUT, DEFAULT_OUTPUT
from src.paths import AFTERSTATE_TWOPLAYER_MODEL, AFTERSTATE_MULTIPLAYER_MODEL
from src.searchstate import SearchBoardState


def fake_state(colours):
    # Small fake server state used by route helpers
    return {
        "players": [{"colour": colour, "name": colour} for colour in colours],
        "pins": {colour: [] for colour in colours},
        "current_turn_colour": colours[0],
        "status": "PLAYING",
    }


def main():
    # Dummy objects let the tests check which model was chosen without loading real models
    base = object()
    multiplayer = object()
    base_search = object()
    multiplayer_search = object()

    # Two player games use the 2p model Larger games use the multiplayer model
    assert player._select_afterstate_agent(fake_state(["red", "blue"]), base, multiplayer) is base
    assert player._select_afterstate_agent(fake_state(["red", "yellow", "lawn green"]), base, multiplayer) is multiplayer
    assert player._select_afterstate_agent(fake_state(["red", "blue", "yellow", "purple"]), base, multiplayer) is multiplayer
    assert player._select_afterstate_agent(
        fake_state(["red", "blue", "yellow", "purple", "gray0"]),
        base,
        multiplayer,
    ) is multiplayer
    assert player._select_afterstate_agent(
        fake_state(["red", "lawn green", "yellow", "blue", "gray0", "purple"]),
        base,
        multiplayer,
    ) is multiplayer
    assert player._select_afterstate_agent(
        fake_state(["red", "blue", "yellow", "purple"]),
        base,
        multiplayer,
        base_search,
        multiplayer_search,
    ) is multiplayer_search
    assert player._select_afterstate_agent(
        fake_state(["red", "blue", "yellow", "purple", "gray0"]),
        base,
        multiplayer,
        base_search,
        multiplayer_search,
    ) is multiplayer_search
    assert player._select_afterstate_agent(
        fake_state(["red", "lawn green", "yellow", "blue", "gray0", "purple"]),
        base,
        multiplayer,
        base_search,
        multiplayer_search,
    ) is multiplayer_search

    # The route must work for every player count used by the server
    assert player.should_use_learned_agent(fake_state(["red", "blue"]), "red")
    assert player.should_use_learned_agent(fake_state(["red", "yellow", "lawn green"]), "red")
    assert player.should_use_learned_agent(fake_state(["red", "blue", "yellow", "purple"]), "red")
    assert player.should_use_learned_agent(fake_state(["red", "blue", "yellow", "purple", "gray0"]), "red")
    assert player.should_use_learned_agent(
        fake_state(["red", "lawn green", "yellow", "blue", "gray0", "purple"]),
        "red",
    )

    # Check the short route names used in README examples and scripts
    # A number at the end changes the switch turn
    assert player._parse_search_settle_switch("search46seatbase") == 75
    assert player._parse_search_settle_switch("search46seatbase80") == 80
    assert player._parse_search_settle_switch("search4corridor6seatbase") == 75
    assert player._parse_search_settle_switch("search4corridor6seatbase80") == 80
    assert player._parse_search_settle_switch("search46seatcorridor") == 75
    assert player._parse_search_settle_switch("search46seatcorridor80") == 80
    assert player._parse_search_settle_switch("search46corridor") == 75
    assert player._parse_search_settle_switch("search46corridor80") == 80
    assert parse_models("search46seatbase,search46seatbase80") == [
        "search46seatbase",
        "search46seatbase80",
    ]
    assert parse_models("search4corridor6seatbase,search4corridor6seatbase80") == [
        "search4corridor6seatbase",
        "search4corridor6seatbase80",
    ]
    assert parse_models("search46seatcorridor,search46seatcorridor80") == [
        "search46seatcorridor",
        "search46seatcorridor80",
    ]
    assert parse_models("search46corridor,search46corridor80") == [
        "search46corridor",
        "search46corridor80",
    ]
    assert parse_models("search46seatmix,search46seatmix80") == [
        "search46seatmix",
        "search46seatmix80",
    ]
    assert parse_models("search46brs,search46seatbrs") == [
        "search46brs",
        "search46seatbrs",
    ]

    # Training uses the same short route names as the benchmark script
    assert parse_mix("greedy,seatbase,seatmix") == [
        "greedy",
        "seatbase",
        "seatmix",
    ]
    assert parse_mix("greedy,seatmix,brs") == [
        "greedy",
        "seatmix",
        "brs",
    ]
    assert parse_mix("greedy,brs,seatbrs") == [
        "greedy",
        "brs",
        "seatbrs",
    ]
    assert parse_mix("greedy,corridor,seatcorridor") == [
        "greedy",
        "corridor",
        "seatcorridor",
    ]
    assert parse_mix("greedy,corridor4seatbase") == [
        "greedy",
        "corridor4seatbase",
    ]

    # Multiplayer training must not replace the 2 player model
    assert DEFAULT_OUTPUT != AFTERSTATE_TWOPLAYER_MODEL
    assert DEFAULT_BEST_OUTPUT != AFTERSTATE_TWOPLAYER_MODEL
    assert AFTERSTATE_MULTIPLAYER_MODEL != AFTERSTATE_TWOPLAYER_MODEL

    # Search must keep the same player order as the GUI and server
    pins = [
        player._PinProxy(0, "blue", 0),
        player._PinProxy(0, "yellow", 1),
        player._PinProxy(0, "purple", 2),
    ]
    state = SearchBoardState.from_board(pins, "red", player._board)
    assert state.player_order == ("red", "blue", "yellow", "purple")
    assert state.next_player_after("red") == "blue"
    assert state.next_player_after("blue") == "yellow"
    assert state.next_player_after("purple") == "red"

    # This test changes tournament player settings then puts them back
    old_brs = player.USE_AFTERSTATE_SEARCH_BRS
    old_seatbrs = player.USE_AFTERSTATE_SEARCH_SEATBRS
    old_seatmix = player.USE_AFTERSTATE_SEARCH_SEATMIX
    player.USE_AFTERSTATE_SEARCH_BRS = True
    player.USE_AFTERSTATE_SEARCH_SEATBRS = False
    player.USE_AFTERSTATE_SEARCH_SEATMIX = False
    try:
        # Put blue partly in target so it is the clear player to chase
        blue_target = sorted(player._target_cells["blue"])[:3]
        brs_order = player._search_player_order_for_state(
            {
                "players": [
                    {"colour": "red", "name": "red"},
                    {"colour": "blue", "name": "blue"},
                    {"colour": "yellow", "name": "yellow"},
                    {"colour": "purple", "name": "purple"},
                ],
                "pins": {
                    "red": [0],
                    "blue": blue_target,
                    "yellow": [1],
                    "purple": [2],
                },
                "current_turn_colour": "red",
                "status": "PLAYING",
            },
            ["red", "blue", "yellow", "purple"],
        )
        # Search should look at red first, then blue, then the rest
        assert brs_order[:2] == ["red", "blue"]
    finally:
        player.USE_AFTERSTATE_SEARCH_BRS = old_brs
        player.USE_AFTERSTATE_SEARCH_SEATBRS = old_seatbrs
        player.USE_AFTERSTATE_SEARCH_SEATMIX = old_seatmix

    # Seat mix should only chase the leader after the chosen switch turn
    old_brs = player.USE_AFTERSTATE_SEARCH_BRS
    old_seatbrs = player.USE_AFTERSTATE_SEARCH_SEATBRS
    old_seatmix = player.USE_AFTERSTATE_SEARCH_SEATMIX
    player.USE_AFTERSTATE_SEARCH_BRS = False
    player.USE_AFTERSTATE_SEARCH_SEATBRS = False
    player.USE_AFTERSTATE_SEARCH_SEATMIX = True
    try:
        # Use the same board, but move past the opening turn limit
        seatmix_state = {
            "players": [
                {"colour": "red", "name": "red"},
                {"colour": "blue", "name": "blue"},
                {"colour": "yellow", "name": "yellow"},
                {"colour": "purple", "name": "purple"},
            ],
            "pins": {
                "red": [0],
                "blue": blue_target,
                "yellow": [1],
                "purple": [2],
            },
            "current_turn_colour": "red",
            "move_count": player.SEARCH_SETTLE_SWITCH_TURN,
            "status": "PLAYING",
        }
        assert player._should_use_best_reply_pressure(seatmix_state)
        seatmix_order = player._search_player_order_for_state(
            seatmix_state,
            ["red", "blue", "yellow", "purple"],
        )
        assert seatmix_order[:2] == ["red", "blue"]
    finally:
        player.USE_AFTERSTATE_SEARCH_BRS = old_brs
        player.USE_AFTERSTATE_SEARCH_SEATBRS = old_seatbrs
        player.USE_AFTERSTATE_SEARCH_SEATMIX = old_seatmix

    print("Routing checks passed.")


if __name__ == "__main__":
    main()

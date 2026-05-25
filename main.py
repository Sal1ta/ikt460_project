# Local entry point for playing games against the trained agents

import argparse
import os
import random
import select
import sys
import time

import numpy as np

from src.afterstate import AfterstateSearchAgent, AfterstateValueAgent
from src.agents import GreedyAgent, MinimaxAgent, RandomAgent
from src.board import HexBoard
from src.env import ChineseCheckersEnv
from src.game import GameManager, RUNNING, TimeoutManager
from src.gui import BoardGUI, choose_game_setup_gui
from src.paths import (
    AFTERSTATE_TWOPLAYER_MODEL,
    AFTERSTATE_FINAL_MODEL,
    AFTERSTATE_TRAINED_MODEL,
    first_existing,
)
from scripts.benchmark import (
    SearchBestReplyRoutingController,
    SearchCorridorRoutingController,
    SearchFourCorridorSeatBaseRoutingController,
    SearchSeatBaseRoutingController,
    SearchSeatBestReplyRoutingController,
    SearchSeatMixRoutingController,
    SearchSettleEarlyRoutingController,
    TournamentFallbackController,
    TournamentRoutingController,
    build_frozen_opponent,
)

np.random.seed()
random.seed()

SUPPORTED_PLAYER_COUNTS = {2, 3, 4, 5, 6}
OPENING_EXPLORATION_TURNS = 12
SELFPLAY_STYLE_MODES = {"asvr", "asvg", "asvm"}
TWO_PLAYER_BASE_COLORS = ["red", "lawn green", "yellow"]
WATCH_CONTROLLER_DEFAULT = "search46seatmix120"
WATCH_OPPONENT_DEFAULT = "afterstate"

# Each menu row keeps the terminal choice, internal mode code, and display label together
GAME_MODE_OPTIONS = (
    ("1", "hvh", "Human vs Human"),
    ("2", "hvrandom", "Human vs Random"),
    ("3", "hvgreedy", "Human vs Greedy"),
    ("4", "gvr", "Greedy vs Random"),
    ("5", "rvr", "Random vs Random"),
    ("6", "gvg", "Greedy vs Greedy"),
    ("7", "hvminimax", "Human vs Minimax"),
    ("8", "mvm", "Minimax vs Minimax"),
    ("9", "hvafter", "Human vs Afterstate"),
    ("10", "asvr", "Afterstate vs Random"),
    ("11", "asvg", "Afterstate vs Greedy"),
    ("12", "asvm", "Afterstate vs Minimax"),
    ("13", "watch", "Watch final tournament route"),
)
GAME_MODE_BY_CHOICE = {choice: mode for choice, mode, _ in GAME_MODE_OPTIONS}
GAME_MODE_LABELS = {mode: label for _, mode, label in GAME_MODE_OPTIONS}
RANDOM_TWO_PLAYER_MODES = {
    "hvh", "hvrandom", "hvgreedy", "gvr", "rvr", "gvg",
    "hvminimax", "mvm", "hvafter", "asvr", "asvg", "asvm",
}
COMPETITION_AFTERSTATE_DEPTH = 2
COMPETITION_AFTERSTATE_ENDGAME_DEPTH = 3
COMPETITION_AFTERSTATE_WIDTH = 8
COMPETITION_AFTERSTATE_ENDGAME_WIDTH = 10
COMPETITION_AFTERSTATE_RESPONSE_WIDTH = 5
COMPETITION_AFTERSTATE_ENDGAME_RESPONSE_WIDTH = 6
COMPETITION_AFTERSTATE_TIME_BUDGET = 0.35

def _configure_competition_afterstate(agent):
    # Local play uses the faster tournament search profile
    if not isinstance(agent, AfterstateSearchAgent):
        return agent

    agent.search_depth = COMPETITION_AFTERSTATE_DEPTH
    agent.endgame_depth = COMPETITION_AFTERSTATE_ENDGAME_DEPTH
    agent.search_width = COMPETITION_AFTERSTATE_WIDTH
    agent.endgame_width = COMPETITION_AFTERSTATE_ENDGAME_WIDTH
    agent.response_width = COMPETITION_AFTERSTATE_RESPONSE_WIDTH
    agent.endgame_response_width = COMPETITION_AFTERSTATE_ENDGAME_RESPONSE_WIDTH
    return agent

def load_afterstate_agent():
    # Saved Afterstate checkpoints use the fixed three layer board view
    state_size = 363
    model_path = first_existing(
        AFTERSTATE_TWOPLAYER_MODEL,
        AFTERSTATE_TRAINED_MODEL,
        AFTERSTATE_FINAL_MODEL,
    )
    if os.path.exists(model_path):
        # Local Afterstate modes use the same value first route as tournament play
        agent = AfterstateValueAgent(
            state_size=state_size,
            player_color="yellow",
            name="AfterstateAgent",
        )
        if agent.load_model(model_path, verbose=False):
            agent.epsilon = 0.0
            return agent

        search_agent = AfterstateSearchAgent(
            state_size=state_size,
            player_color="yellow",
            name="AfterstateSearchAgent",
        )
        if search_agent.load_model(model_path, verbose=False):
            search_agent.epsilon = 0.0
            _configure_competition_afterstate(search_agent)
            return search_agent

        print("Afterstate checkpoint is incompatible with the current code.")
    else:
        print("No Afterstate checkpoint found. Using default weights.")
    return AfterstateValueAgent(state_size=state_size, player_color="yellow", name="AfterstateAgent")

def check_win(pins_on_board, player_color, board):
    opposite_color = board.colour_opposites.get(player_color)
    if not opposite_color:
        return False
    target_cells = {i for i, cell in enumerate(board.cells) if cell.postype == opposite_color}
    return sum(1 for p in pins_on_board if p.color == player_color and p.axialindex in target_cells) == 10

def is_reverse_action(action, current_player, pins_on_board, recent_moves):
    pin_id, dest = action
    pin = next((p for p in pins_on_board if p.color == current_player and p.id == pin_id), None)
    if pin is None:
        return False

    old_pos = pin.axialindex
    for colour, old_pin_id, old_cell, new_cell in recent_moves[-12:]:
        if colour == current_player and old_pin_id == pin_id and old_cell == dest and new_cell == old_pos:
            return True
    return False

def remember_move(recent_moves, current_player, pin_id, old_cell, new_cell):
    recent_moves.append((current_player, int(pin_id), old_cell, new_cell))
    if len(recent_moves) > 30:
        recent_moves.pop(0)

def _mode_player_labels(game_mode):
    labels = {
        "hvh": ("Human 1", "Human 2"),
        "hvrandom": ("Human", "Random"),
        "hvgreedy": ("Human", "Greedy"),
        "gvr": ("Greedy", "Random"),
        "rvr": ("Random", "Random"),
        "gvg": ("Greedy", "Greedy"),
        "hvminimax": ("Human", "Minimax"),
        "mvm": ("Minimax", "Minimax"),
        "hvafter": ("Afterstate", "Human"),
        "asvr": ("Afterstate", "Random"),
        "asvg": ("Afterstate", "Greedy"),
        "asvm": ("Afterstate", "Minimax"),
    }
    return labels.get(game_mode)

def build_player_roles(game_mode, player_colors):
    labels = _mode_player_labels(game_mode)
    if labels is None:
        return {}
    if len(player_colors) > 2:
        first_role, opponent_role = labels
        return {
            str(colour): first_role if index == 0 else opponent_role
            for index, colour in enumerate(player_colors)
        }
    return {
        str(player_colors[index]): labels[index]
        for index in range(min(len(player_colors), len(labels)))
    }

def _sample_two_player_colours(game):
    base_colour = random.choice(TWO_PLAYER_BASE_COLORS)
    colours = [base_colour, game.board.colour_opposites[base_colour]]
    random.shuffle(colours)
    return colours

def _announce_side_assignment(game_mode, colours):
    labels = _mode_player_labels(game_mode)
    if labels is None:
        return
    first_label, second_label = labels
    print(f"Controllers this game: {first_label.upper()}={colours[0].upper()}, {second_label.upper()}={colours[1].upper()}.")
    print(f"Turn order this game: {colours[0].upper()} moves first, {colours[1].upper()} moves second.")

def configure_player_colors_for_mode(game, game_mode, player_colors):
    # Two player automated games reshuffle the opposite colour lane and starting side
    if game.num_players == 2 and game_mode in RANDOM_TWO_PLAYER_MODES:
        colours = _sample_two_player_colours(game)
        game.sync_player_state(colours)
        print("This two-player mode uses a random opposite-colour lane each game.")
        print("Both the lane and the side assignment are reshuffled on every start.")
        print(f"Active lane this game: {colours[0].upper()} vs {colours[1].upper()}.")
        _announce_side_assignment(game_mode, colours)
        return colours

    return player_colors

def _get_ai_type(game_mode, current_player, player_colors):
    # The mode string maps each colour to a controller type
    p0, p1 = player_colors[0], player_colors[1]
    if len(player_colors) > 2:
        first = player_colors[0]
        rest = player_colors[1:]
        multi_rules = {
            "hvafter":  {first: "afterstate"},
            "asvr":     {first: "afterstate", **{colour: "random" for colour in rest}},
            "asvg":     {first: "afterstate", **{colour: "greedy" for colour in rest}},
            "asvm":     {first: "afterstate", **{colour: "minimax" for colour in rest}},
            "hvrandom": {colour: "random" for colour in rest},
            "hvgreedy": {colour: "greedy" for colour in rest},
            "hvminimax":{colour: "minimax" for colour in rest},
            "gvr":      {first: "greedy", **{colour: "random" for colour in rest}},
            "rvr":      {colour: "random" for colour in player_colors},
            "gvg":      {colour: "greedy" for colour in player_colors},
            "mvm":      {colour: "minimax" for colour in player_colors},
        }
        return multi_rules.get(game_mode, {}).get(current_player)

    rules = {
        "hvafter":  {p0: "afterstate"},
        "asvr":     {p0: "afterstate", p1: "random"},
        "asvg":     {p0: "afterstate", p1: "greedy"},
        "asvm":     {p0: "afterstate", p1: "minimax"},
        "hvrandom": {p1: "random"},
        "hvgreedy": {p1: "greedy"},
        "hvminimax":{p1: "minimax"},
        "gvr":      {p0: "greedy",    p1: "random"},
        "rvr":      {p0: "random",    p1: "random"},
        "gvg":      {p0: "greedy",    p1: "greedy"},
        "mvm":      {p0: "minimax",   p1: "minimax"},
    }
    return rules.get(game_mode, {}).get(current_player)

def _select_ai_action(ai_type, agents, pins_on_board, current_player, remaining, board, explore_opening=False, turn_count=None, move_history=None):
    # Only the learned agents use extra opening exploration to vary their openings
    if ai_type == "afterstate":
        return agents["afterstate"].choose_action_from_board(
            pins_on_board,
            current_player,
            remaining,
            board,
            explore=explore_opening,
            time_budget_seconds=COMPETITION_AFTERSTATE_TIME_BUDGET,
            turn_count=turn_count,
            move_history=move_history,
        )
    if ai_type == "greedy":
        return agents["greedy"].choose_action_from_board(pins_on_board, current_player, remaining, board)
    if ai_type == "minimax":
        return agents["minimax"].choose_action_from_board(pins_on_board, current_player, remaining, board)
    if ai_type == "random":
        return agents["random"].choose_action(None, remaining)
    return None

def _parse_optional_int(raw, default):
    raw = str(raw or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Could not parse {raw!r} as a number; using {default}.")
        return default

def _supported_player_text():
    return ", ".join(str(count) for count in sorted(SUPPORTED_PLAYER_COUNTS))

def _prompt_start_command():
    raw = input("Type 'start' to begin, or 'exit' to quit: ").strip().lower()
    if raw in {"exit", "quit", "q"}:
        return "exit"
    if raw in {"start", "s"}:
        return "start"
    return None

def _prompt_player_count():
    num_players = _parse_optional_int(input(f"Enter number of players ({_supported_player_text()}): "), 0)
    if num_players not in SUPPORTED_PLAYER_COUNTS:
        print(f"Invalid player count. Choose one of: {_supported_player_text()}.")
        return None
    return num_players

def _prompt_game_mode():
    print("Choose game mode:")
    for choice, _, label in GAME_MODE_OPTIONS:
        print(f"  {choice}. {label}")

    mode_input = input(f"Enter number (1-{len(GAME_MODE_OPTIONS)}): ").strip().lower()
    game_mode = GAME_MODE_BY_CHOICE.get(mode_input)
    if game_mode is None:
        print(f"Invalid mode. Choose 1-{len(GAME_MODE_OPTIONS)}.")
    return game_mode

def _normalise_watch_kind(kind, default):
    kind = str(kind or "").strip().lower()
    return kind or default

def _normalise_watch_controller_kind(kind):
    kind = _normalise_watch_kind(kind, WATCH_CONTROLLER_DEFAULT)
    aliases = {
        "s": "search46seatbase",
        "sb": "search46seatbase",
        "seat": "search46seatbase",
        "default": WATCH_CONTROLLER_DEFAULT,
        "locked": WATCH_CONTROLLER_DEFAULT,
        "final": WATCH_CONTROLLER_DEFAULT,
        "a": "afterstate",
        "as": "afterstate",
        "t": "tournament",
        "f": "fallback",
        "g": "search46brs",
        "br": "search46brs",
        "mix": WATCH_CONTROLLER_DEFAULT,
        "lane": "search4corridor6seatbase",
        "corridor": "search4corridor6seatbase",
        "sc": "search4corridor6seatbase",
        "combo": "search4corridor6seatbase",
        "lane4": "search4corridor6seatbase",
        "4lane": "search4corridor6seatbase",
    }
    return aliases.get(kind, kind)

def _normalise_watch_opponent_kind(kind):
    kind = _normalise_watch_kind(kind, WATCH_OPPONENT_DEFAULT)
    aliases = {
        "a": "afterstate",
        "as": "afterstate",
        "c": "challenger",
        "ch": "challenger",
        "o": "original",
        "base": "original",
        "g": "greedy",
        "r": "random",
        "m": "minimax",
        "f": "fallback",
        "s": "seatbase",
        "sb": "seatbase",
        "default": WATCH_OPPONENT_DEFAULT,
    }
    return aliases.get(kind, kind)

def _normalise_watch_model_path(raw):
    raw = str(raw or "").strip()
    if raw.lower() in {"", "q", "quit", "none", "no", "default", "-"}:
        return None
    if not os.path.exists(raw):
        print(f"Checkpoint {raw!r} does not exist; using the current locked model.")
        return None
    return raw

def _watch_label(kind, model_path=None):
    raw = str(kind).strip()
    label = {
        "afterstate": "Afterstate tournament route",
        "tournament": "Afterstate tournament route",
        "search": "Afterstate search route",
        "fallback": "Handcrafted fallback",
        "greedy": "Greedy",
        "random": "Random",
    }.get(raw, raw)

    for prefix, display, default_turn in (
        ("search46seatmix", "Afterstate Search SeatMix", 75),
        ("seatmix", "Afterstate Search SeatMix", 75),
        ("search46seatbase", "Afterstate Search SeatBase", 75),
        ("seatbase", "Afterstate Search SeatBase", 75),
        ("search46settle", "Afterstate Search Settle", 60),
        ("settle", "Afterstate Search Settle", 60),
        ("search46corridor", "Afterstate Search Corridor", 75),
        ("corridor", "Afterstate Search Corridor", 75),
        ("search46seatcorridor", "Afterstate Search SeatCorridor", 75),
        ("seatcorridor", "Afterstate Search SeatCorridor", 75),
    ):
        if raw == prefix:
            label = f"{display} {default_turn}"
            break
        if raw.startswith(prefix) and raw[len(prefix):].isdigit():
            label = f"{display} {raw[len(prefix):]}"
            break

    if model_path:
        label = f"{label} ({os.path.basename(str(model_path))})"
    return label

def _suffix_for_prefixes(kind, prefixes):
    for prefix in prefixes:
        if kind == prefix:
            return ""
        if kind.startswith(prefix) and kind[len(prefix):].isdigit():
            return kind[len(prefix):]
    return None

def _make_watch_controller(kind, model_path=None):
    kind = _normalise_watch_controller_kind(kind)
    if kind in ("afterstate", "tournament"):
        return TournamentRoutingController(model_path=model_path)
    if kind == "search":
        return TournamentRoutingController(model_path=model_path, search_agent=True)
    if kind == "fallback":
        return TournamentFallbackController()
    seatbase_suffix = _suffix_for_prefixes(kind, ("search46seatbase", "seatbase", "latebase"))
    if kind == "auto" or seatbase_suffix is not None:
        return SearchSeatBaseRoutingController(
            model_path=model_path,
            base_model_path=AFTERSTATE_TWOPLAYER_MODEL if model_path else None,
            switch_turn=int(seatbase_suffix) if seatbase_suffix else 75,
        )
    settle_suffix = _suffix_for_prefixes(kind, ("search46settle", "settle", "duelcycle"))
    if settle_suffix is not None:
        return SearchSettleEarlyRoutingController(
            model_path=model_path,
            switch_turn=int(settle_suffix) if settle_suffix else 75,
            min_players=4,
        )
    seatmix_suffix = _suffix_for_prefixes(kind, ("search46seatmix", "seatmix"))
    if seatmix_suffix is not None:
        switch_turn = int(seatmix_suffix) if seatmix_suffix else 75
        return SearchSeatMixRoutingController(
            model_path=model_path,
            base_model_path=AFTERSTATE_TWOPLAYER_MODEL if model_path else None,
            switch_turn=switch_turn,
            turn_limit=max(90, switch_turn),
        )
    corridor_suffix = _suffix_for_prefixes(kind, ("search46corridor", "corridor"))
    if corridor_suffix is not None:
        return SearchCorridorRoutingController(
            SearchSettleEarlyRoutingController(
                model_path=model_path,
                switch_turn=int(corridor_suffix) if corridor_suffix else 75,
                min_players=4,
            )
        )
    seatcorridor_suffix = _suffix_for_prefixes(kind, ("search46seatcorridor", "seatcorridor"))
    if seatcorridor_suffix is not None:
        return SearchCorridorRoutingController(
            SearchSeatBaseRoutingController(
                model_path=model_path,
                base_model_path=AFTERSTATE_TWOPLAYER_MODEL if model_path else None,
                switch_turn=int(seatcorridor_suffix) if seatcorridor_suffix else 75,
            ),
            seat_safe=True,
        )
    combo_suffix = _suffix_for_prefixes(kind, ("search4corridor6seatbase", "corridor4seatbase", "lane4seat6"))
    if combo_suffix is not None:
        return SearchFourCorridorSeatBaseRoutingController(
            model_path=model_path,
            base_model_path=AFTERSTATE_TWOPLAYER_MODEL if model_path else None,
            switch_turn=int(combo_suffix) if combo_suffix else 75,
        )
    if kind in ("search46brs", "brs"):
        return SearchBestReplyRoutingController(model_path=model_path, min_players=4)
    if kind in ("search46seatbrs", "seatbrs"):
        return SearchSeatBestReplyRoutingController(model_path=model_path)
    raise ValueError(
        f"Unsupported watch controller {kind!r}. "
        "Try search46seatbase, lane, combo, search4corridor6seatbase, afterstate, search46seatmix, search46brs, or fallback."
    )

def _make_watch_opponent(kind):
    kind = _normalise_watch_opponent_kind(kind)
    if kind == "greedy":
        return GreedyAgent(name="Greedy")
    if kind == "random":
        return RandomAgent(name="Random")
    if kind == "minimax":
        return MinimaxAgent(depth=2)
    return build_frozen_opponent(kind)

def _choose_watch_action(agent, env, valid_actions, move_count):
    if isinstance(agent, (
        TournamentRoutingController,
        TournamentFallbackController,
        SearchSettleEarlyRoutingController,
        SearchSeatBaseRoutingController,
        SearchSeatMixRoutingController,
        SearchCorridorRoutingController,
        SearchFourCorridorSeatBaseRoutingController,
        SearchBestReplyRoutingController,
        SearchSeatBestReplyRoutingController,
    )):
        return agent.choose_action(env, valid_actions, move_count)
    if hasattr(agent, "choose_action"):
        return agent.choose_action(env, valid_actions)
    return None

def _print_watch_scoreboard(env):
    rows = []
    for colour in env.player_colors:
        pins = env.count_player_pins_in_target(colour)
        distance = env.total_distance_to_target(colour)
        score = pins * 100.0 + max(0.0, 400.0 - 2.0 * distance)
        rows.append((score, colour, pins, distance))
    rows.sort(reverse=True)
    print("\nScoreboard:")
    for rank, (score, colour, pins, distance) in enumerate(rows, start=1):
        print(f"  {rank}. {colour.upper():10s} pins={pins:2d} distance={distance:4.0f} score={score:6.1f}")

def watch_models(
    num_players=6,
    controller_kind=WATCH_CONTROLLER_DEFAULT,
    opponent_kind=WATCH_OPPONENT_DEFAULT,
    model_path=None,
    seed=None,
    max_turns=300,
    clock_limit_seconds=None,
    delay=0.08,
):
    if num_players not in SUPPORTED_PLAYER_COUNTS:
        raise ValueError("watch mode supports 2, 3, 4, 5, or 6 players")
    controller_kind = _normalise_watch_controller_kind(controller_kind)
    opponent_kind = _normalise_watch_opponent_kind(opponent_kind)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    controller = _make_watch_controller(controller_kind, model_path=model_path)
    opponent = _make_watch_opponent(opponent_kind)

    env = ChineseCheckersEnv(num_players=num_players, max_turns=max_turns)
    env.reset()
    if clock_limit_seconds is None:
        clock_limit_seconds = 60.0 * num_players
    clock_limit_seconds = float(clock_limit_seconds)
    watched_colour = str(env.player_colors[0])
    controller_label = _watch_label(controller_kind, model_path)
    opponent_label = _watch_label(opponent_kind)
    player_roles = {
        str(colour): controller_label if str(colour) == watched_colour else opponent_label
        for colour in env.player_colors
    }

    print("\nWatch mode")
    print(f"  players:    {num_players}")
    print(f"  controller: {controller_label} as {watched_colour.upper()}")
    print(f"  opponents:  {opponent_label}")
    print(f"  max turns:  {max_turns}")
    if clock_limit_seconds > 0:
        print(f"  clock stop: {clock_limit_seconds:.0f}s tournament estimate")
    print("  close the GUI window to stop early\n")

    gui = BoardGUI(env.board, env.pins_on_board, player_roles=player_roles)
    gui.window.update()

    info = {}
    watch_started_at = time.time()
    while not env.done and env.turn_count < max_turns:
        if clock_limit_seconds > 0 and time.time() - watch_started_at >= clock_limit_seconds:
            info = {"message": f"Tournament clock estimate reached ({clock_limit_seconds:.0f}s)"}
            break
        try:
            current = str(env.get_current_player())
            valid_actions = env.get_valid_actions()
            gui.set_turn(env.turn_count + 1)
            gui._update_active_dot(current)
            gui.set_status(f"{current.upper()} is thinking...")
            gui.window.update()
        except Exception:
            print("Watch window closed.")
            return

        if not valid_actions:
            print(f"{current.upper()} has no valid moves. Skipping.")
            env.game.move_manager.next_player_turn()
            continue

        active_agent = controller if current == watched_colour else opponent
        action = _choose_watch_action(active_agent, env, valid_actions, env.turn_count)
        if action is None:
            print(f"{current.upper()} did not return a move. Stopping watch.")
            break

        pin_id, dest_id = action
        players_pin = env.game.get_pin_id_of_player(env.pins_on_board, current, str(pin_id))
        if players_pin is None:
            print(f"{current.upper()} returned invalid pin {pin_id}. Stopping watch.")
            break

        old_cell = players_pin.axialindex
        move_path = players_pin.get_move_path(dest_id)
        _, _, done, info = env.step(action)
        print(f"Turn {env.turn_count:3d}: {current.upper():10s} pin {pin_id} {old_cell}->{dest_id}")

        try:
            gui.animate_move(
                env.pins_on_board,
                int(pin_id),
                current,
                move_path,
                status_msg=f"{current.upper()} moved pin {pin_id} -> {dest_id}",
            )
            if delay > 0:
                time.sleep(delay)
        except Exception:
            print("Watch window closed.")
            return

        if done:
            break

    message = info.get("message", "Max turns reached")
    print(f"\nWatch finished: {message}")
    winner = message.removesuffix(" wins") if message.endswith(" wins") else None
    if winner:
        try:
            gui.show_winner(winner)
        except Exception:
            pass
    else:
        gui.set_status(message)
        gui.window.update()
    _print_watch_scoreboard(env)
    print("Close the GUI window to return to the menu.")
    try:
        gui.window.mainloop()
    except Exception:
        pass

def _run_watch_from_cli(argv):
    if len(argv) < 2 or argv[1].lower() not in {"watch", "watch-models", "models"}:
        return False
    parser = argparse.ArgumentParser(description="Watch trained Chinese Checkers models play in the GUI.")
    parser.add_argument("watch")
    parser.add_argument("--players", type=int, default=6, choices=sorted(SUPPORTED_PLAYER_COUNTS))
    parser.add_argument(
        "--controller",
        default=WATCH_CONTROLLER_DEFAULT,
        help="Controller to watch. Use lane/combo for 4p corridor + 6p seatbase, or --model to override the selected Afterstate checkpoint.",
    )
    parser.add_argument("--opponent", default=WATCH_OPPONENT_DEFAULT)
    parser.add_argument("--model", default=None, help="Optional checkpoint path for the watched controller.")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument(
        "--clock-limit",
        type=float,
        default=None,
        help="Seconds before watch stops for tournament-clock scoring; default is 60*players, use 0 to disable.",
    )
    parser.add_argument("--delay", type=float, default=0.08)
    args = parser.parse_args(argv[1:])
    watch_models(
        num_players=args.players,
        controller_kind=_normalise_watch_controller_kind(args.controller),
        opponent_kind=_normalise_watch_opponent_kind(args.opponent),
        model_path=_normalise_watch_model_path(args.model),
        seed=_parse_optional_int(args.seed, None),
        max_turns=args.max_turns,
        clock_limit_seconds=args.clock_limit,
        delay=args.delay,
    )
    return True

def _use_gui_setup(argv):
    return len(argv) > 1 and str(argv[1]).strip().lower() in {
        "gui",
        "menu",
        "setup",
        "--gui",
        "--menu",
    }

if __name__ == "__main__":
    if _run_watch_from_cli(sys.argv):
        sys.exit(0)

    use_gui_setup = _use_gui_setup(sys.argv)

    print("Chinese Checkers")
    print()

    while RUNNING:
        if use_gui_setup:
            setup = choose_game_setup_gui(SUPPORTED_PLAYER_COUNTS, GAME_MODE_OPTIONS)
            if setup is None:
                break
            num_players = setup["players"]
            game_mode = setup["mode"]
        else:
            command = _prompt_start_command()

            if command == "exit":
                break

            if command != "start":
                print("Please type 'start' or 'exit'.")
                print()
                continue

            num_players = _prompt_player_count()
            if num_players is None:
                print()
                continue

            game_mode = _prompt_game_mode()
            if game_mode is None:
                print()
                continue

        if game_mode == "watch":
            print("Opening watch mode: final route vs Afterstate.")
            try:
                watch_models(
                    num_players=num_players,
                    controller_kind=WATCH_CONTROLLER_DEFAULT,
                    opponent_kind=WATCH_OPPONENT_DEFAULT,
                    max_turns=300,
                    delay=0.08,
                )
            except Exception as exc:
                print(f"Could not start watch mode: {exc}")
            print()
            continue

        board     = HexBoard(R=4, hole_radius=16, spacing=34)
        num_turns = 0
        game          = GameManager(board, num_players=num_players)
        player_colors = game.assign_players_colors(num_players)

        print(f"Mode: {GAME_MODE_LABELS.get(game_mode, game_mode)}")

        player_colors = configure_player_colors_for_mode(game, game_mode, player_colors)
        player_roles = build_player_roles(game_mode, player_colors)
        pins_on_board = game.place_pins_to_board(player_colors)

        print("\nPlayers: " + ", ".join(c.upper() for c in player_colors))
        if player_roles:
            print("Controllers: " + ", ".join(f"{colour.upper()}={role}" for colour, role in player_roles.items()))
        print()
        board.print_ascii(pins=pins_on_board, empty="·")

        agents = {
            "afterstate": load_afterstate_agent() if game_mode in ("hvafter", "asvr", "asvg", "asvm") else None,
            "random":     RandomAgent()           if game_mode in ("hvrandom", "gvr", "rvr", "asvr") else None,
            "greedy":     GreedyAgent()           if game_mode in ("hvgreedy", "gvr", "gvg", "asvg") else None,
            "minimax":    MinimaxAgent(depth=2)   if game_mode in ("hvminimax", "mvm", "asvm") else None,
        }

        gui = BoardGUI(board, pins_on_board, player_roles=player_roles)
        gui.window.update()
        game.timeout_manager = TimeoutManager(game.player_colors, turn_time_limit=60, game_time_limit=1800)
        game.timeout_manager.start_game_timer()

        MAX_TURNS           = 500
        board_state_history = {}
        recent_moves        = []

        while True:
            current_player = game.move_manager.get_current_player()
            game.timeout_manager.start_turn()
            gui.set_turn(num_turns + 1)
            gui._update_active_dot(current_player)

            print(f"\nTurn {num_turns + 1}: {current_player.upper()}")

            if num_turns >= MAX_TURNS:
                print(f"Max turns ({MAX_TURNS}) reached. Game over.")
                break

            # Local play stops visible repetition loops quickly
            state_key = tuple(sorted((p.color, p.axialindex) for p in pins_on_board))
            board_state_history[state_key] = board_state_history.get(state_key, 0) + 1
            if board_state_history[state_key] >= 3:
                print("Board state repeated 3 times. Draw.")
                break

            if game.timeout_manager.is_turn_timeout():
                print(f"{current_player.upper()} timed out. Turn skipped.")
                game.timeout_manager.end_turn(current_player)
                game.move_manager.next_player_turn()
                break

            if game.timeout_manager.is_game_timeout(current_player):
                print(f"{current_player.upper()} exceeded total game time. Game over.")
                game.timeout_manager.end_game_timer()
                break

            # Automated turns share one flow gather legal moves, filter
            # reversals, choose an action
            ai_type = _get_ai_type(game_mode, current_player, player_colors)

            if ai_type is not None:
                gui.set_status(f"{current_player.upper()} is thinking...")
                gui.window.update()
                occupied      = {pin.axialindex for pin in pins_on_board}
                valid_actions = [(pin.id, dest)
                                 for pin in pins_on_board if pin.color == current_player
                                 for dest in pin.get_legal_moves() if dest not in occupied]

                if not valid_actions:
                    print(f"{current_player.upper()} has no valid moves. Skipping.")
                    game.timeout_manager.end_turn(current_player)
                    game.move_manager.next_player_turn()
                    continue

                move_pin_success = False
                attempted = set()
                while not move_pin_success and len(attempted) < len(valid_actions):
                    remaining = [a for a in valid_actions if a not in attempted]

                    # Local automated play avoids immediate reversals when alternatives exist
                    safe_remaining = [a for a in remaining if not is_reverse_action(a, current_player, pins_on_board, recent_moves)]
                    if safe_remaining:
                        remaining = safe_remaining

                    explore_opening = (
                        game_mode in SELFPLAY_STYLE_MODES
                        and ai_type == "afterstate"
                        and num_turns < OPENING_EXPLORATION_TURNS
                    )
                    action = _select_ai_action(
                        ai_type,
                        agents,
                        pins_on_board,
                        current_player,
                        remaining,
                        board,
                        explore_opening=explore_opening,
                        turn_count=num_turns,
                        move_history=recent_moves,
                    )

                    if action is None:
                        break
                    attempted.add(action)
                    pin_id, dest_id  = action
                    players_pin      = game.get_pin_id_of_player(pins_on_board, current_player, str(pin_id))
                    if players_pin is None:
                        continue
                    old_cell         = players_pin.axialindex
                    move_path        = players_pin.get_move_path(dest_id)
                    move_pin_success = players_pin.place_pin(dest_id)

                if move_pin_success:
                    remember_move(recent_moves, current_player, pin_id, old_cell, dest_id)
                    print(f"  {current_player.upper()} moved pin {pin_id} -> cell {dest_id}")
                    board.print_ascii(pins=pins_on_board, empty='·')
                    gui.animate_move(
                        pins_on_board,
                        pin_id,
                        current_player,
                        move_path,
                        status_msg=f"{current_player.upper()} moved pin {pin_id} -> {dest_id}",
                    )
                    time_used = game.timeout_manager.end_turn(current_player)
                    game.move_manager.log_move(current_player, f"({pin_id},{dest_id})", players_pin, dest_id, time_used)
                    if check_win(pins_on_board, current_player, board):
                        num_turns += 1
                        print(f"\n{current_player.upper()} WINS after {num_turns} turns!")
                        gui.show_winner(current_player)
                        break
                    game.move_manager.next_player_turn()
                    num_turns += 1
                else:
                    print(f"{current_player.upper()} could not complete a move. Skipping.")
                    game.timeout_manager.end_turn(current_player)
                    game.move_manager.next_player_turn()
                continue

            # Human input and GUI clicks both become the same action tuple
            gui.enable_click(current_player)
            print("Click a pin, or type: pin_id,dest_id  |  pass  |  exit")
            print("> ", end='', flush=True)

            move_input    = None
            pin_id_click  = None
            dest_id_click = None

            while move_input is None and pin_id_click is None:
                gui.window.update()
                if gui._pending_action is not None:
                    pin_id_click, dest_id_click = gui._pending_action
                    gui._pending_action = None
                    break
                try:
                    if select.select([sys.stdin], [], [], 0)[0]:
                        move_input = sys.stdin.readline().strip()
                except Exception:
                    pass
            gui.disable_click()

            if pin_id_click is not None:
                print(f"({pin_id_click},{dest_id_click})")
                players_pin      = game.get_pin_id_of_player(pins_on_board, current_player, str(pin_id_click))
                if players_pin is None:
                    print("Invalid pin.")
                    continue
                old_cell         = players_pin.axialindex
                move_path        = players_pin.get_move_path(dest_id_click)
                move_pin_success = players_pin.place_pin(dest_id_click)
                if move_pin_success:
                    remember_move(recent_moves, current_player, pin_id_click, old_cell, dest_id_click)
                    print(f"  {current_player.upper()} moved pin {pin_id_click} -> cell {dest_id_click}")
                    board.print_ascii(pins=pins_on_board, empty='·')
                    gui.animate_move(
                        pins_on_board,
                        pin_id_click,
                        current_player,
                        move_path,
                        status_msg=f"You moved pin {pin_id_click} -> {dest_id_click}",
                    )
                    time_used = game.timeout_manager.end_turn(current_player)
                    game.move_manager.log_move(current_player, f"({pin_id_click},{dest_id_click})", players_pin, dest_id_click, time_used)
                    if check_win(pins_on_board, current_player, board):
                        num_turns += 1
                        print(f"\n{current_player.upper()} WINS after {num_turns} turns!")
                        gui.show_winner(current_player)
                        break
                    game.move_manager.next_player_turn()
                    num_turns += 1
                else:
                    print("Invalid move, please try again.")
                continue

            if not move_input:
                move_input = ''

            if move_input.lower() == 'exit':
                break

            if move_input.lower() == 'pass':
                print(f"  {current_player.upper()} passes.")
                time_used = game.timeout_manager.end_turn(current_player)
                game.move_manager.log_move(current_player, "pass", None, None, time_used)
                game.move_manager.next_player_turn()
                continue

            if ',' not in move_input or len(move_input.split(',')) != 2:
                print("Invalid format. Use: pin_id,dest_id")
                continue

            if game.timeout_manager.is_turn_timeout():
                print(f"{current_player.upper()} timed out. Turn skipped.")
                time_used = game.timeout_manager.end_turn(current_player)
                game.move_manager.log_move(current_player, "timeout", None, None, time_used)
                game.move_manager.next_player_turn()
                continue

            pin_num = move_input.split(',')[0].replace('(', '')
            try:
                dest_id = int(move_input.split(',')[1].replace(')', ''))
            except ValueError:
                print("Invalid destination.")
                continue

            players_pin      = game.get_pin_id_of_player(pins_on_board, current_player, pin_num)
            if players_pin is None:
                print("Invalid pin.")
                continue
            old_cell         = players_pin.axialindex
            move_path        = players_pin.get_move_path(dest_id)
            move_pin_success = players_pin.place_pin(dest_id)

            if move_pin_success:
                remember_move(recent_moves, current_player, players_pin.id, old_cell, dest_id)
                print(f"  {current_player.upper()} moved pin {pin_num} -> cell {dest_id}")
                board.print_ascii(pins=pins_on_board, empty='·')
                gui.animate_move(pins_on_board, players_pin.id, current_player, move_path)
                time_used = game.timeout_manager.end_turn(current_player)
                game.move_manager.log_move(current_player, move_input, players_pin, dest_id, time_used)
                if check_win(pins_on_board, current_player, board):
                    num_turns += 1
                    print(f"\n{current_player.upper()} WINS after {num_turns} turns!")
                    gui.show_winner(current_player)
                    break
                game.move_manager.next_player_turn()
                num_turns += 1
            else:
                print("Invalid move, please try again.")

        print(f"\nGame over. {num_turns} turns played.")
        print("\nMove history:")
        for color, moves in game.move_manager.move_history.items():
            print(f"\n{color.upper()} ({len(moves)} moves):")
            for move in moves:
                print(f"  {move}")

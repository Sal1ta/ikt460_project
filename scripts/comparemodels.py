# Compare tournament routes

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import GreedyAgent, RandomAgent
from src.paths import (
    AFTERSTATE_TWOPLAYER_MODEL,
    AFTERSTATE_MULTIPLAYER_MODEL,
    OUTPUTS_DIR,
)
from scripts.benchmark import (
    SearchBestReplyRoutingController,
    SearchCorridorRoutingController,
    SearchFourCorridorSeatBaseRoutingController,
    SearchSeatBaseRoutingController,
    SearchSeatMixRoutingController,
    SearchSeatBestReplyRoutingController,
    SearchSettleEarlyRoutingController,
    SUPPORTED_OPPONENT_KINDS,
    TournamentFallbackController,
    TournamentRoutingController,
    build_frozen_opponent,
    notify_finished,
    run_one_match,
    seed_everything,
)


DEFAULT_SUMMARY_CSV = OUTPUTS_DIR / "comparisonsummary.csv"
DEFAULT_RAW_CSV = OUTPUTS_DIR / "comparisongames.csv"


def parse_csv_ints(raw, allowed=None):
    # Turn comma separated input into checked numbers before games start
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if allowed is not None and value not in allowed:
            raise ValueError(f"{value} is not in allowed set {sorted(allowed)}")
        values.append(value)
    if not values:
        raise ValueError("at least one value is required")
    return values


def parse_models(raw, extra_labels=None):
    # These are the short route names used in tables, README, and commands
    extra_labels = set(extra_labels or [])
    models = [token.strip().lower() for token in str(raw).split(",") if token.strip()]
    allowed = {
        # Keep these names stable because docs and scripts use them
        "original",
        "challenger",
        "auto",
        "fallback",
        "search",
        "search46brs",
        "search46seatbase",
        "search46corridor",
        "search46seatcorridor",
        "search4corridor6seatbase",
        "search46seatmix",
        "search46seatbrs",
        "search46settle",
    }
    invalid = [
        # Accept names ending in a number, such as search46seatmix120
        model for model in models
        if model not in allowed
        and not (model.startswith("search46seatbase") and model.removeprefix("search46seatbase").isdigit())
        and not (model.startswith("search46corridor") and model.removeprefix("search46corridor").isdigit())
        and not (model.startswith("search46seatcorridor") and model.removeprefix("search46seatcorridor").isdigit())
        and not (model.startswith("search4corridor6seatbase") and model.removeprefix("search4corridor6seatbase").isdigit())
        and not (model.startswith("search46seatmix") and model.removeprefix("search46seatmix").isdigit())
        and not (model.startswith("search46settle") and model.removeprefix("search46settle").isdigit())
        and model not in extra_labels
    ]
    if invalid:
        raise ValueError(f"unknown model labels: {invalid}")
    if not models:
        raise ValueError("at least one model label is required")
    return models


def parse_extra_models(raw):
    # Extra models let us test one model file without changing the code
    extra = {}
    if not raw:
        return extra
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError("extra models must use label=path format")
        label, path = token.split("=", 1)
        label = label.strip().lower()
        if not label:
            raise ValueError("extra model labels cannot be empty")
        # The optional route name tells the script how to run the model
        route, model_path = parse_extra_model_spec(path)
        extra[label] = {"route": route, "path": model_path}
    return extra


def is_extra_route(route):
    # A route name tells the script which controller to use with the model file
    return (
        route in {
            "raw",
            "tournament",
            "afterstate",
            "auto",
            "seatbase",
            "search46seatbase",
            "corridor",
            "search46corridor",
            "seatcorridor",
            "search46seatcorridor",
            "corridor4seatbase",
            "search4corridor6seatbase",
            "seatmix",
            "search46seatmix",
            "settle",
            "search46settle",
            "brs",
            "search46brs",
            "seatbrs",
            "search46seatbrs",
        }
        or (route.startswith("search46seatbase") and route.removeprefix("search46seatbase").isdigit())
        or (route.startswith("search46corridor") and route.removeprefix("search46corridor").isdigit())
        or (route.startswith("search46seatcorridor") and route.removeprefix("search46seatcorridor").isdigit())
        or (route.startswith("search4corridor6seatbase") and route.removeprefix("search4corridor6seatbase").isdigit())
        or (route.startswith("search46seatmix") and route.removeprefix("search46seatmix").isdigit())
        or (route.startswith("search46settle") and route.removeprefix("search46settle").isdigit())
    )


def parse_extra_model_spec(raw):
    # Accept either a path only or routepath
    spec = raw.strip()
    route = "raw"
    model_path = spec
    if ":" in spec:
        maybe_route, maybe_path = spec.split(":", 1)
        maybe_route = maybe_route.strip().lower()
        if is_extra_route(maybe_route):
            # Use the part before as a route only if it is a known route name
            route = maybe_route
            model_path = maybe_path.strip()
    if not model_path:
        raise ValueError("extra model path cannot be empty")
    return route, Path(model_path)


def make_extra_controller(spec):
    # Extra controllers use the same rules as the final routes
    route = spec["route"]
    model_path = spec["path"]
    if route in ("raw", "tournament", "afterstate"):
        return TournamentRoutingController(model_path=model_path)
    if route in ("auto", "seatbase", "search46seatbase"):
        return SearchSeatBaseRoutingController(
            model_path=model_path,
            base_model_path=AFTERSTATE_TWOPLAYER_MODEL,
        )
    if route.startswith("search46seatbase") and route.removeprefix("search46seatbase").isdigit():
        return SearchSeatBaseRoutingController(
            model_path=model_path,
            base_model_path=AFTERSTATE_TWOPLAYER_MODEL,
            switch_turn=int(route.removeprefix("search46seatbase")),
        )
    if route in ("corridor", "search46corridor"):
        return SearchCorridorRoutingController(
            SearchSettleEarlyRoutingController(model_path=model_path, switch_turn=75),
        )
    if route.startswith("search46corridor") and route.removeprefix("search46corridor").isdigit():
        return SearchCorridorRoutingController(
            SearchSettleEarlyRoutingController(
                model_path=model_path,
                switch_turn=int(route.removeprefix("search46corridor")),
            ),
        )
    if route in ("seatcorridor", "search46seatcorridor"):
        return SearchCorridorRoutingController(
            SearchSeatBaseRoutingController(
                model_path=model_path,
                base_model_path=AFTERSTATE_TWOPLAYER_MODEL,
            ),
            seat_safe=True,
        )
    if route.startswith("search46seatcorridor") and route.removeprefix("search46seatcorridor").isdigit():
        return SearchCorridorRoutingController(
            SearchSeatBaseRoutingController(
                model_path=model_path,
                base_model_path=AFTERSTATE_TWOPLAYER_MODEL,
                switch_turn=int(route.removeprefix("search46seatcorridor")),
            ),
            seat_safe=True,
        )
    if route in ("corridor4seatbase", "search4corridor6seatbase"):
        return SearchFourCorridorSeatBaseRoutingController(
            model_path=model_path,
            base_model_path=AFTERSTATE_TWOPLAYER_MODEL,
            switch_turn=75,
        )
    if route.startswith("search4corridor6seatbase") and route.removeprefix("search4corridor6seatbase").isdigit():
        return SearchFourCorridorSeatBaseRoutingController(
            model_path=model_path,
            base_model_path=AFTERSTATE_TWOPLAYER_MODEL,
            switch_turn=int(route.removeprefix("search4corridor6seatbase")),
        )
    if route in ("seatmix", "search46seatmix"):
        return SearchSeatMixRoutingController(
            model_path=model_path,
            base_model_path=AFTERSTATE_TWOPLAYER_MODEL,
        )
    if route.startswith("search46seatmix") and route.removeprefix("search46seatmix").isdigit():
        switch_turn = int(route.removeprefix("search46seatmix"))
        return SearchSeatMixRoutingController(
            model_path=model_path,
            base_model_path=AFTERSTATE_TWOPLAYER_MODEL,
            switch_turn=switch_turn,
            turn_limit=max(90, switch_turn),
        )
    if route in ("settle", "search46settle"):
        return SearchSettleEarlyRoutingController(model_path=model_path)
    if route.startswith("search46settle") and route.removeprefix("search46settle").isdigit():
        return SearchSettleEarlyRoutingController(
            model_path=model_path,
            switch_turn=int(route.removeprefix("search46settle")),
        )
    if route in ("brs", "search46brs"):
        return SearchBestReplyRoutingController(model_path=model_path)
    if route in ("seatbrs", "search46seatbrs"):
        return SearchSeatBestReplyRoutingController(model_path=model_path)
    raise ValueError(f"unsupported extra model route: {route}")


def make_controller(model_name, extra_models=None):
    # Map each printed model name to the controller used in local games
    extra_models = extra_models or {}
    if model_name == "original":
        # Original means only the 2 player model
        return TournamentRoutingController(model_path=AFTERSTATE_TWOPLAYER_MODEL)
    if model_name == "challenger":
        # Challenger means only the multiplayer model
        return TournamentRoutingController(model_path=AFTERSTATE_MULTIPLAYER_MODEL)
    if model_name == "auto":
        # Auto means the final route 2p model plus multiplayer model
        return SearchSeatBaseRoutingController()
    if model_name == "search":
        return TournamentRoutingController(search_agent=True)
    if model_name == "search46settle":
        return SearchSettleEarlyRoutingController()
    if model_name.startswith("search46settle") and model_name.removeprefix("search46settle").isdigit():
        return SearchSettleEarlyRoutingController(
            switch_turn=int(model_name.removeprefix("search46settle")),
        )
    if model_name == "search46seatbase":
        return SearchSeatBaseRoutingController()
    if model_name == "search46corridor":
        return SearchCorridorRoutingController(
            SearchSettleEarlyRoutingController(switch_turn=75),
        )
    if model_name == "search46seatcorridor":
        return SearchCorridorRoutingController(
            SearchSeatBaseRoutingController(switch_turn=75),
            seat_safe=True,
        )
    if model_name == "search4corridor6seatbase":
        return SearchFourCorridorSeatBaseRoutingController(switch_turn=75)
    if model_name == "search46seatmix":
        return SearchSeatMixRoutingController()
    if model_name == "search46brs":
        return SearchBestReplyRoutingController()
    if model_name == "search46seatbrs":
        return SearchSeatBestReplyRoutingController()
    if model_name.startswith("search46seatbase") and model_name.removeprefix("search46seatbase").isdigit():
        return SearchSeatBaseRoutingController(
            switch_turn=int(model_name.removeprefix("search46seatbase")),
        )
    if model_name.startswith("search46corridor") and model_name.removeprefix("search46corridor").isdigit():
        return SearchCorridorRoutingController(
            SearchSettleEarlyRoutingController(
                switch_turn=int(model_name.removeprefix("search46corridor")),
            ),
        )
    if model_name.startswith("search46seatcorridor") and model_name.removeprefix("search46seatcorridor").isdigit():
        return SearchCorridorRoutingController(
            SearchSeatBaseRoutingController(
                switch_turn=int(model_name.removeprefix("search46seatcorridor")),
            ),
            seat_safe=True,
        )
    if model_name.startswith("search4corridor6seatbase") and model_name.removeprefix("search4corridor6seatbase").isdigit():
        return SearchFourCorridorSeatBaseRoutingController(
            switch_turn=int(model_name.removeprefix("search4corridor6seatbase")),
        )
    if model_name.startswith("search46seatmix") and model_name.removeprefix("search46seatmix").isdigit():
        switch_turn = int(model_name.removeprefix("search46seatmix"))
        return SearchSeatMixRoutingController(
            switch_turn=switch_turn,
            turn_limit=max(90, switch_turn),
        )
    if model_name == "fallback":
        return TournamentFallbackController()
    if model_name in extra_models:
        return make_extra_controller(extra_models[model_name])
    raise ValueError(f"unsupported model: {model_name}")


def make_opponent_factory(kind):
    # Simple opponents are made per colour Learned opponents are loaded once
    if kind == "greedy":
        return lambda colour: GreedyAgent(name=f"Greedy-{colour}")
    if kind == "random":
        return lambda colour: RandomAgent(name=f"Random-{colour}")
    frozen = build_frozen_opponent(kind)
    return lambda colour: frozen


def summarize(results, num_players):
    # Turn game results into the summary rows written to CSV
    games = len(results)
    wins = sum(1 for result in results if result["my_won"])
    avg_rank = sum(result["my_rank"] for result in results) / games
    avg_pins = sum(result["my_pins"] for result in results) / games
    avg_distance = sum(result["my_distance"] for result in results) / games
    avg_score = sum(result["my_score"] for result in results) / games
    rank_hist = {
        # Rank counts show if a model is usually near the top or often fails
        rank: sum(1 for result in results if result["my_rank"] == rank)
        for rank in range(1, num_players + 1)
    }
    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games * 100.0,
        "avg_rank": avg_rank,
        "avg_pins": avg_pins,
        "avg_distance": avg_distance,
        "avg_score": avg_score,
        "rank_histogram": " ".join(f"{rank}:{count}" for rank, count in rank_hist.items()),
    }


def combined_score(summary_rows):
    # Small score used to compare models quickly Raw numbers stay in the CSV
    total = 0.0
    for row in summary_rows:
        num_players = int(row["players"])
        neutral_rank = (num_players + 1) / 2.0
        total += (
            float(row["win_rate"])
            + float(row["avg_pins"]) * 8.0
            + (neutral_rank - float(row["avg_rank"])) * 15.0
        )
    return total


def write_csv(path, rows, fieldnames):
    # Make output folders here so the script works from a clean project copy
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Use fixed seeds, player counts, and seats so results are easier to compare
    parser = argparse.ArgumentParser(description="Compare tournament-route models across fixed seeds.")
    parser.add_argument("--players", default="3,4,6", help="Comma-separated player counts from 3,4,6.")
    parser.add_argument("--seeds", default="20260521,20260522", help="Comma-separated random seeds.")
    parser.add_argument("--games", type=int, default=72)
    parser.add_argument("--opponent", default="greedy", choices=SUPPORTED_OPPONENT_KINDS)
    parser.add_argument("--models", default="original,challenger,auto")
    parser.add_argument(
        "--extra-models",
        default="",
        help="Optional label=path or label=route:path entries, comma-separated.",
    )
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--raw-csv", default=str(DEFAULT_RAW_CSV))
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    player_counts = parse_csv_ints(args.players, allowed={2, 3, 4, 5, 6})
    seeds = parse_csv_ints(args.seeds)
    extra_models = parse_extra_models(args.extra_models)
    models = parse_models(args.models, extra_labels=extra_models)
    for label in extra_models:
        if label not in models:
            # Add extra models automatically after their path is given once
            models.append(label)
    opponent_factory = make_opponent_factory(args.opponent)

    summary_rows = []
    raw_rows = []
    start = time.time()

    print("\nModel comparison")
    print(f"  models:   {models}")
    print(f"  players:  {player_counts}")
    print(f"  seeds:    {seeds}")
    print(f"  games:    {args.games} per seed/player/model")
    print(f"  opponent: {args.opponent}\n")

    for model_name in models:
        # Build the controller once so every game uses the same loaded models
        controller = make_controller(model_name, extra_models=extra_models)
        model_summary_rows = []
        for num_players in player_counts:
            for seed in seeds:
                seed_everything(seed)
                # Rotate seats so one colour does not decide the whole result
                results = [
                    run_one_match(
                        controller,
                        num_players,
                        opponent_factory,
                        max_turns=args.max_turns,
                        agent_seat=game_index % num_players,
                    )
                    for game_index in range(args.games)
                ]
                stats = summarize(results, num_players)
                # Summary rows give one line per model, player count, and seed
                summary_row = {
                    "model": model_name,
                    "players": num_players,
                    "seed": seed,
                    **stats,
                }
                summary_rows.append(summary_row)
                model_summary_rows.append(summary_row)

                print(
                    f"  {model_name:10s} {num_players}p seed={seed} "
                    f"wins={stats['wins']:2d}/{stats['games']} "
                    f"rank={stats['avg_rank']:.2f} pins={stats['avg_pins']:.2f} "
                    f"score={stats['avg_score']:.1f}"
                )

                for game_index, result in enumerate(results, start=1):
                    # Raw rows keep full game details for plots and checks
                    raw_rows.append({
                        "model": model_name,
                        "players": num_players,
                        "seed": seed,
                        "game": game_index,
                        "seat": result["agent_seat"],
                        "colour": result["my_colour"],
                        "rank": result["my_rank"],
                        "won": int(result["my_won"]),
                        "pins": result["my_pins"],
                        "distance": result["my_distance"],
                        "score": result["my_score"],
                        "total_moves": result["total_moves"],
                        "my_moves": result["my_moves"],
                        "result": result["result"],
                    })

        print(f"  combined {model_name:10s}: {combined_score(model_summary_rows):.2f}\n")

    summary_fields = [
        "model",
        "players",
        "seed",
        "games",
        "wins",
        "win_rate",
        "avg_rank",
        "avg_pins",
        "avg_distance",
        "avg_score",
        "rank_histogram",
    ]
    raw_fields = [
        "model",
        "players",
        "seed",
        "game",
        "seat",
        "colour",
        "rank",
        "won",
        "pins",
        "distance",
        "score",
        "total_moves",
        "my_moves",
        "result",
    ]
    write_csv(args.summary_csv, summary_rows, summary_fields)
    write_csv(args.raw_csv, raw_rows, raw_fields)

    print(f"Saved summary CSV: {args.summary_csv}")
    print(f"Saved raw game CSV: {args.raw_csv}")
    print(f"Elapsed: {time.time() - start:.0f}s")
    if args.notify:
        notify_finished(message="Model comparison finished")


if __name__ == "__main__":
    main()

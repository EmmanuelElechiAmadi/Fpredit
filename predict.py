"""
Usage:
    python predict.py "Arsenal" "Chelsea" --league EPL --data-dir data/raw
    python predict.py --demo               # runs on generated synthetic data, no download needed
"""

import argparse
import hashlib
import pickle
import warnings
from pathlib import Path

from src.data_loader import generate_synthetic_league, load_league_csvs
from src.ensemble import FootballEnsemble


def _source_hash() -> str:
    """Hash of all source files in src/ used as a cache invalidation key.
    If any source code changes, cached models are automatically invalidated."""
    src_dir = Path(__file__).parent / "src"
    hasher = hashlib.sha256()
    for path in sorted(src_dir.glob("*.py")):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def get_model(league: str, data_dir: str, demo: bool, cache_dir="models"):
    cache_path = Path(cache_dir) / f"{league}_{_source_hash()}.pkl"

    # Clean up stale cache files for this league (old hash versions)
    if not demo:
        for stale in Path(cache_dir).glob(f"{league}_*.pkl"):
            if stale != cache_path:
                stale.unlink(missing_ok=True)

    if cache_path.exists() and not demo:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    df = generate_synthetic_league() if demo else load_league_csvs(data_dir, league)
    model = FootballEnsemble()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        model.fit(df)

    if not demo:
        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(model, f)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "home", nargs="?", help="Home team name (must match data source spelling)"
    )
    ap.add_argument("away", nargs="?", help="Away team name")
    ap.add_argument("--league", default="EPL", choices=["EPL", "LALIGA", "SERIEA"])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic data (no CSV download needed)",
    )
    args = ap.parse_args()

    if args.demo and not args.home:
        # demo mode with no teams given -> just show it works with a sample matchup
        args.home, args.away = "Team A", "Team B"

    if not args.home or not args.away:
        ap.error("Provide home and away team names, or use --demo for a sample run")

    model = get_model(args.league, args.data_dir, args.demo)
    result = model.predict(args.home, args.away)

    print(f"\n{args.home} vs {args.away} ({args.league})")
    print("-" * 50)
    print(f"Home win: {result['home_win']:.1%}")
    print(f"Draw:     {result['draw']:.1%}")
    print(f"Away win: {result['away_win']:.1%}")
    print(
        f"\nMost likely scoreline: {result['most_likely_score'][0]}-{result['most_likely_score'][1]}"
    )
    print(
        f"Expected goals: {result['expected_goals'][0]:.2f} - {result['expected_goals'][1]:.2f}"
    )
    print(f"Over 2.5 goals: {result['over_2_5_goals']:.1%}")
    print(f"Both teams to score: {result['btts_yes']:.1%}")
    print(
        f"\n(component breakdown - Dixon-Coles H/D/A: {[f'{x:.1%}' for x in result['component_probs']['dixon_coles']]})"
    )
    print(
        f"(component breakdown - Elo H/D/A:          {[f'{x:.1%}' for x in result['component_probs']['elo']]})"
    )


if __name__ == "__main__":
    main()

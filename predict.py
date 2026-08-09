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

from src.config import load_config
from src.data_loader import generate_synthetic_league, load_league_csvs
from src.model_factory import build_ensemble


def _source_hash() -> str:
    """Hash of all source files in src/ used as a cache invalidation key.
    If any source code changes, cached models are automatically invalidated."""
    src_dir = Path(__file__).parent / "src"
    hasher = hashlib.sha256()
    for path in sorted(src_dir.glob("*.py")):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def get_model(
    league: str,
    data_dir: str,
    demo: bool,
    cache_dir="models",
    cfg=None,
    xg_df=None,
):
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
    model = build_ensemble(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        model.fit(df, xg_df=xg_df)

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
        "--xg-dir",
        default=None,
        help="Directory of cached Understat xG CSVs (data/xg) to enable xG features.",
    )
    ap.add_argument(
        "--odds",
        nargs=3,
        type=float,
        metavar=("HOME_ODDS", "DRAW_ODDS", "AWAY_ODDS"),
        default=None,
        help="Optional decimal closing odds, e.g. --odds 2.1 3.4 3.6 (enables "
        "the market-residual layer for this prediction).",
    )
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

    cfg = load_config()
    xg_df = None
    if args.xg_dir:
        from src.xg_loader import load_league_xg

        xg_df = load_league_xg(args.xg_dir, args.league)
        print(f"Loaded xG for {args.league} ({len(xg_df)} matches)")
    model = get_model(args.league, args.data_dir, args.demo, cfg=cfg, xg_df=xg_df)
    result = model.predict(
        args.home, args.away, market_odds=tuple(args.odds) if args.odds else None
    )

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
    print(
        f"(component breakdown - Dynamic H/D/A:      {[f'{x:.1%}' for x in result['component_probs']['dynamic']]})"
    )
    if "xg" in result["component_probs"]:
        print(
            f"(component breakdown - xG H/D/A:          {[f'{x:.1%}' for x in result['component_probs']['xg']]})"
        )
    if "market_home" in model.feature_cols:
        print(
            "\nMarket-residual layer active (model was trained with market features; "
            "pass --odds to supply the closing line for this fixture)"
        )


if __name__ == "__main__":
    main()

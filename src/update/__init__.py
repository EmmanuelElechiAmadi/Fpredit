"""League Updates pipeline.

SQLite-backed orchestration that keeps results, match stats, xG, odds and
fixtures current for every league the model covers. It reuses the statistical
engines in ``src/`` (Elo, Dixon-Coles, features, xG loader, fixtures, market)
and only adds the fetch / normalise / validate / store / report layer.

Run it with::

    python -m src.update.run --leagues EPL LALIGA SERIEA
    python -m src.update.run --leagues EPL --since 2026-11-01
    python -m src.update.run --dry-run
"""

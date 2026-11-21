"""Run report — a Markdown file summarising one update run (Section 10)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _pct(x: Optional[float]) -> str:
    return f"{x:.1%}" if isinstance(x, (int, float)) else "—"


def build_report(results: dict) -> str:
    lines: list[str] = []
    started = results.get("started_at", "")
    finished = results.get("finished_at", "")
    lines.append(f"# League Updates — {started[:10]}")
    lines.append("")
    lines.append(f"- Started: {started}")
    lines.append(f"- Finished: {finished}")
    lines.append(f"- Season: {results.get('season', '—')}")
    if results.get("dry_run"):
        lines.append("- Mode: **dry-run** (validate only, nothing written)")
    lines.append("")

    halted = results.get("halted", [])
    if halted:
        lines.append("## Halted leagues")
        lines.append("")
        for league, reason in halted:
            lines.append(f"- **{league}**: {reason}")
        lines.append("")

    for league, r in results.get("leagues", {}).items():
        lines.append(f"## {league} — {r.get('status', '?')}")
        lines.append("")
        lines.append(
            f"- Matches: **{r.get('added', 0)} added**, "
            f"**{r.get('corrected', 0)} corrected**, "
            f"{r.get('unchanged', 0)} unchanged, "
            f"**{r.get('quarantined', 0)} quarantined** "
            f"({r.get('row_count', 0)} stored)"
        )
        lines.append(f"- Odds snapshots inserted: {r.get('odds_inserted', 0)}")
        lines.append(
            f"- Latest match date: {r.get('latest_match_date') or '—'}  |  "
            f"xG: {r.get('xg_complete', 0)} complete, {r.get('xg_pending', 0)} pending"
        )

        wf = r.get("walk_forward") or {}
        if wf.get("n"):
            lines.append(
                f"- Walk-forward log-loss: model **{wf['log_loss_model']:.4f}** "
                f"vs market **{_ll(wf)}** "
                f"(n={wf['n']}, {'beats' if _beats(wf) else 'trails'} market)"
            )

        movers = r.get("elo_movers") or []
        if movers:
            lines.append("")
            lines.append("### Largest Elo changes")
            lines.append("")
            lines.append("| Team | Old | New | Δ |")
            lines.append("|---|---:|---:|---:|")
            for m in movers[:10]:
                old = f"{m['old']:.1f}" if m.get("old") is not None else "—"
                lines.append(
                    f"| {m['team']} | {old} | {m['new']:.1f} | {m['delta']:+.1f} |"
                )

        standings = r.get("standings") or []
        if standings:
            lines.append("")
            lines.append("### League table (computed from results)")
            lines.append("")
            lines.append("| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Form |")
            lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
            for s in standings:
                lines.append(
                    f"| {s['rank']} | {s['team']} | {s['played']} | {s['won']} "
                    f"| {s['draw']} | {s['lost']} | {s['gf']} | {s['ga']} "
                    f"| {s['goal_diff']:+d} | {s['points']} | {s.get('form', '')} |"
                )

        preds = r.get("predictions") or []
        if preds:
            lines.append("")
            lines.append("### Upcoming fixtures")
            lines.append("")
            lines.append("| Date | Home | Away | H | D | A | Pick | Edge |")
            lines.append("|---|---|---:|---:|---:|---|---:|")
            for p in preds[:40]:
                edge = f"{p['edge']:+.1%}" if p.get("edge") is not None else "—"
                pick = p.get("pick") or "—"
                lines.append(
                    f"| {p.get('date', '—')} | {p['home']} | {p['away']} "
                    f"| {_pct(p['p_home'])} | {_pct(p['p_draw'])} | {_pct(p['p_away'])} "
                    f"| {pick} | {edge} |"
                )

        warnings = r.get("warnings") or []
        if warnings:
            lines.append("")
            lines.append("### Warnings")
            for w in warnings:
                lines.append(f"- {w}")
        lines.append("")

    global_warnings = results.get("warnings") or []
    if global_warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in global_warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _ll(wf: dict) -> str:
    v = wf.get("log_loss_market")
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def _beats(wf: dict) -> bool:
    m, mk = wf.get("log_loss_model"), wf.get("log_loss_market")
    return isinstance(m, (int, float)) and isinstance(mk, (int, float)) and m < mk


def write_report(markdown: str, reports_dir: str | Path) -> Path:
    path = Path(reports_dir)
    path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    out = path / f"update_{ts}.md"
    out.write_text(markdown)
    return out

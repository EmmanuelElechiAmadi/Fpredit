# data/fixtures — 2026-27 placeholder schedules

The season CSVs in this folder (`E0/2026-27.csv`, `SP1/2026-27.csv`,
`I1/2026-27.csv`) are **programmatically generated placeholder round-robins**,
not the official fixture lists. They exist so the Fixtures view works from day
one and the matchweek calendar (GW1…GW38, dates one week apart from each
league's real opening Saturday) lines up with the actual season.

Pairings are synthetic (circle method over the league's current roster — see
`scripts/generate_placeholder_fixtures.py`), so they will **not** match the real
schedule's home/away assignments or postponements.

## When the official file is available

Replace the placeholder wholesale:

```bash
# football-data.co.uk layout (Date,HomeTeam,AwayTeam,Div,B365H,...) or the
# minimal layout are both understood — just drop it in:
cp E0.csv data/fixtures/E0/2026-27.csv
```

The loader picks the file up automatically. Rows whose dates have passed stop
showing in the *upcoming* fixtures view on their own (dates are filtered), so a
stale placeholder never masquerades as a live prediction — but once real
results land in `data/raw/`, keep the results CSV and the snapshot flow in sync
as described in the project README.

Regenerate all placeholders (e.g. after editing a roster) with:

```bash
python3 scripts/generate_placeholder_fixtures.py
```

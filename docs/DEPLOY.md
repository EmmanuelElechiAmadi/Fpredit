# Deployment guide

The app is a persistent FastAPI service with CPU-heavy model fits and baked-in
data. **Vercel (serverless) is a poor fit** — function size/time limits and the
lack of a persistent process break the model warm-up and backtests. Use a
platform that runs a long-lived container instead.

The repo is deploy-ready: a `Dockerfile`, `render.yaml` and the ~3MB of results +
xG data are committed so the image builds without re-downloading anything.

## Option A — Railway (recommended: persistent, simplest)

1. Push this repo to GitHub (already done).
2. Go to https://railway.app → **New Project** → **Deploy from GitHub repo** → pick `Fpredit`.
3. Railway auto-detects the `Dockerfile` (via `railway.json`). It builds and starts.
4. Open the generated URL (e.g. `https://<project>.up.railway.app`).

Railway runs the process persistently, so the startup warm-up (~3-4 min) runs
once and every view is fast afterwards.

## Option B — Render (free tier, but spins down when idle)

1. Push the repo to GitHub.
2. https://render.com → **New** → **Web Service** → connect the repo.
3. Render reads `render.yaml` (runtime: docker). Deploy.
4. First load after ~15 min idle triggers a cold start (~1-2 min, the warm-up
   runs then) — fine for a demo, not for continuous use.

## Option C — Hugging Face Spaces (free, ML-friendly)

1. Create a Space at https://huggingface.co/new-space → choose **Docker** runtime.
2. Push the repo to the Space (or import from GitHub in the Space settings).
3. The `Dockerfile` is used automatically. The app is served and can be embedded
   in an iframe.

## Manual / other platforms (Fly.io, Cloud Run)

```bash
docker build -t football-predictor .
docker run -p 8000:8000 football-predictor   # serves http://localhost:8000
```
- **Fly.io:** `fly launch` then `fly deploy` (uses the Dockerfile).
- **Cloud Run:** `gcloud run deploy football-predictor --source . --port 8000`.

## What happens on deploy

- The image contains: app code, web UI, config, and the football data.
- On first boot uvicorn starts and the app pre-warms the model caches and the
  default backtest in the background (`app/main.py` `_warm_caches`).
- Health endpoint: `GET /api/health` → `{"status":"ok"}` (used by platform
  health checks).

## Scheduled data refresh (keep the deployed app fresh)

The container's entrypoint (`docker/entrypoint.sh`) runs the weekly refresh
(`scripts/refresh_data.py`) instead of the web server when `CRON_REFRESH=1` is
set — so a scheduled job can reuse the exact same image.

- **GitHub Actions** (easiest, no platform config): `.github/workflows/weekly-refresh.yml`
  runs `python3 scripts/refresh_data.py --seasons 1 --strict` every Monday and
  commits any changed `data/` back to `master`. Enable it in the repo settings →
  Actions → Workflow permissions (read **and write**). For automatic standings
  + fixtures, add a `FOOTBALL_DATA_ORG_API_KEY` repository secret.
- **Render**: `render.yaml` already defines a `cron` service
  (`football-predictor-weekly-refresh`, Mondays 03:00 UTC) with
  `CRON_REFRESH=1`. Fill in `FOOTBALL_DATA_ORG_API_KEY` on the dashboard.
- **Railway**: New → **Cron Job** → deploy this repo, then set the environment
  variable `CRON_REFRESH=1` (Railway runs the container on schedule; the
  entrypoint executes the refresh and exits). Optional:
  `FOOTBALL_DATA_ORG_API_KEY` for automatic standings/fixtures.
- Any other scheduler can just run `python3 scripts/refresh_data.py --seasons 1`
  in a checkout of the repo.

The refreshed container keeps the new `data/` baked into the image only after a
redeploy; the GitHub Actions route commits data to `master`, which triggers the
web service's auto-deploy (a fresh image with the updated CSVs).

## Notes

- The model hyperparameters in `config.yaml` (tuned on real EPL data) are baked
  in — no environment variables needed.
- If you later add more seasons, re-commit `data/` and redeploy (or mount a
  volume); the image is rebuilt with the new CSVs.
- `reports/`, `models/` and log files are intentionally not deployed.

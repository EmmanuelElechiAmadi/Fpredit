# Deployment guide

The app is a persistent FastAPI service with CPU-heavy model fits and baked-in
data. **Vercel (serverless) is a poor fit** — function size/time limits and the
lack of a persistent process break the model warm-up and backtests. Use a
platform that runs a long-lived container instead.

The repo is deploy-ready: a `Dockerfile`, `railway.json`, `render.yaml` and
`Procfile` are included, and the ~3MB of results + xG data is committed so the
image builds without re-downloading anything.

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

## Notes

- The model hyperparameters in `config.yaml` (tuned on real EPL data) are baked
  in — no environment variables needed.
- If you later add more seasons, re-commit `data/` and redeploy (or mount a
  volume); the image is rebuilt with the new CSVs.
- `reports/`, `models/` and log files are intentionally not deployed.

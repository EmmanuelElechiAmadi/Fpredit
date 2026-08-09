.PHONY: install install-dev editable test test-quick test-slow test-coverage clean clean-all lint format format-check typecheck download-data scrape-xg scrape-xg-all calibrate-elo calibrate-model demo predict predict-xg backtest backtest-xg backtest-save report web

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-dev:
	pip install pre-commit
	python3 -m pre_commit install

editable:
	pip install -e .

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	python3 -m pytest tests/ -v --timeout=300

test-quick:
	python3 -m pytest tests/ -x -k "not slow" --timeout=120

test-slow:
	python3 -m pytest tests/ -v -k "slow" --timeout=600

test-coverage:
	python3 -m pytest tests/ --cov=src --cov-report=term-missing --timeout=300 --runslow

# ── Linting & Formatting ──────────────────────────────────────────────────────

lint:
	python3 -m pre_commit run ruff --all-files

format:
	python3 -m pre_commit run black --all-files
	python3 -m pre_commit run isort --all-files

format-check:
	python3 -m pre_commit run black --all-files
	python3 -m pre_commit run isort --all-files

typecheck:
	mypy src/ --ignore-missing-imports

# ── Data ──────────────────────────────────────────────────────────────────────

download-data:
	python3 scripts/download_data.py --all --seasons 5

scrape-xg:
	python3 scripts/scrape_understat.py --league EPL --seasons 5

scrape-xg-all:
	python3 scripts/scrape_understat.py --league EPL --seasons 5
	python3 scripts/scrape_understat.py --league LALIGA --seasons 5
	python3 scripts/scrape_understat.py --league SERIEA --seasons 5

# ── Calibration ────────────────────────────────────────────────────────────────

calibrate-elo:
	python scripts/calibrate_elo_draw.py --demo

calibrate-model:
	python3 scripts/calibrate_model.py --league EPL --xg-dir data/xg --grid quick

calibrate-model-write:
	python3 scripts/calibrate_model.py --league EPL --xg-dir data/xg --grid quick --write-config

# ── Run ───────────────────────────────────────────────────────────────────────

demo:
	python predict.py --demo

predict:
	python predict.py --league EPL

predict-xg:
	python predict.py --league EPL --xg-dir data/xg

backtest:
	python backtest.py --demo

backtest-xg:
	python backtest.py --league EPL --xg-dir data/xg

backtest-save:
	python backtest.py --demo --output backtest_results.csv

report:
	python3 scripts/report.py --league EPL

web:
	python3 -m uvicorn app.main:app --reload --port 8000

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache build dist *.egg-info

clean-all: clean
	rm -rf models/
	rm -rf logs/
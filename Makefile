.PHONY: install install-dev editable test test-quick test-slow test-coverage clean clean-all lint format format-check typecheck download-data calibrate-elo demo predict backtest backtest-save report

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-dev:
	pip install pre-commit
	pre-commit install

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
	pre-commit run ruff --all-files

format:
	pre-commit run black --all-files
	pre-commit run isort --all-files

format-check:
	pre-commit run black --all-files
	pre-commit run isort --all-files

typecheck:
	mypy src/ --ignore-missing-imports

# ── Data ──────────────────────────────────────────────────────────────────────

download-data:
	python3 scripts/download_data.py --all --seasons 5

# ── Calibration ────────────────────────────────────────────────────────────────

calibrate-elo:
	python scripts/calibrate_elo_draw.py --demo

# ── Run ───────────────────────────────────────────────────────────────────────

demo:
	python predict.py --demo

predict:
	python predict.py --league EPL

backtest:
	python backtest.py --demo

backtest-save:
	python backtest.py --demo --output backtest_results.csv

report:
	python3 scripts/report.py --league EPL

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache build dist *.egg-info

clean-all: clean
	rm -rf models/
	rm -rf logs/
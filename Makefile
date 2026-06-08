# Makefile — Phase F shortcuts
# Usage: make <command>

.PHONY: help install test coverage load-test lint clean run

# ── Default ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Phase F — Available Commands"
	@echo "  ────────────────────────────────────────"
	@echo "  make install      Install all dependencies"
	@echo "  make run          Start the Phase E server"
	@echo "  make test         Run all tests"
	@echo "  make coverage     Run tests with coverage report"
	@echo "  make load-test    Run Locust load test (headless)"
	@echo "  make load-ui      Run Locust with browser UI"
	@echo "  make lint         Run code linter"
	@echo "  make clean        Remove cache and report files"
	@echo ""

# ── Install ───────────────────────────────────────────────────────────────────
install:
	pip install -r requirements_e.txt
	pip install pytest-cov coverage locust ruff

# ── Run server ────────────────────────────────────────────────────────────────
run:
	python -m uvicorn app_phase_e:app --port 8001 --reload

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest test_coverage.py -v -m "not load"

test-all:
	pytest -v -m "not load"

# ── Coverage ──────────────────────────────────────────────────────────────────
coverage:
	pytest test_coverage.py \
		--cov=. \
		--cov-report=term-missing \
		--cov-report=html:coverage_report \
		--cov-fail-under=80 \
		-m "not load"
	@echo ""
	@echo "  Coverage report: coverage_report/index.html"
	@echo "  Open it in your browser to see line-by-line coverage"

# ── Load testing ──────────────────────────────────────────────────────────────
load-test:
	locust -f load_test.py \
		--headless \
		--users 50 \
		--spawn-rate 5 \
		--run-time 60s \
		--host http://localhost:8001 \
		--only-summary

load-ui:
	@echo "Open http://localhost:8089 in your browser"
	locust -f load_test.py --host http://localhost:8001

# ── Lint ──────────────────────────────────────────────────────────────────────
lint:
	ruff check . --ignore E501,F401

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf coverage_report .coverage coverage.xml .pytest_cache
	@echo "Cleaned up cache and report files"

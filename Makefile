PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install run test lint format docker-up docker-down ingest-sample coverage

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt -r requirements-dev.txt

run:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest tests/ -q

coverage:
	$(PYTHON) -m pytest tests/ -q --cov --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check app tests scripts
	$(PYTHON) -m black --check app tests scripts

format:
	$(PYTHON) -m ruff check --fix app tests scripts
	$(PYTHON) -m black app tests scripts

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

ingest-sample:
	$(PYTHON) scripts/ingest_samples.py

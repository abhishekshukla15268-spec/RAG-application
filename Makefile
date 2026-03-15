.PHONY: setup start-ui test-eval ingest

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	docker compose up -d

start-ui:
	.venv/bin/streamlit run app.py

ingest:
	.venv/bin/python -m src.engine.ingestion

test-eval:
	.venv/bin/pytest tests/test_rag.py -v

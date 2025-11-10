# Makefile for DL Trading with MLOps

.PHONY: help install verify fetch-data engineer-features train monitor backtest api test-api clean mlflow-ui

help:
	@echo "Available commands:"
	@echo "  make install           - Install all dependencies"
	@echo "  make verify            - Verify installation"
	@echo "  make fetch-data        - Fetch market data"
	@echo "  make engineer-features - Engineer features"
	@echo "  make train             - Train CNN model"
	@echo "  make monitor           - Monitor data drift"
	@echo "  make backtest          - Run backtesting"
	@echo "  make api               - Start API server"
	@echo "  make test-api          - Test API endpoints"
	@echo "  make mlflow-ui         - Start MLFlow UI"
	@echo "  make pipeline          - Run complete pipeline"
	@echo "  make clean             - Clean generated files"

install:
	pip install -r requirements.txt

verify:
	python scripts/verify_installation.py

fetch-data:
	python scripts/01_fetch_data.py

engineer-features:
	python scripts/02_engineer_features.py

train:
	python scripts/03_train_model.py

monitor:
	python scripts/05_monitor_drift.py

backtest:
	python scripts/06_run_backtest.py

api:
	python scripts/04_run_api.py

test-api:
	python scripts/test_api.py

mlflow-ui:
	mlflow ui

pipeline: fetch-data engineer-features train monitor backtest
	@echo "Pipeline completed successfully!"

clean:
	rm -rf data/ models/ reports/ mlruns/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned generated files"

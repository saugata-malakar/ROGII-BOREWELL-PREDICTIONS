# Makefile for ROGII Wellbore Geology Prediction

.PHONY: help setup install clean train predict submit test lint format

help:
	@echo "Available commands:"
	@echo "  make setup      - Create virtual environment and install dependencies"
	@echo "  make install    - Install dependencies only"
	@echo "  make clean      - Remove generated files and cache"
	@echo "  make train      - Train the model"
	@echo "  make predict    - Generate predictions"
	@echo "  make submit     - Validate submission file"
	@echo "  make test       - Run tests (if available)"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code with black"
	@echo "  make notebook   - Launch Jupyter notebook"

setup:
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  Windows: venv\\Scripts\\activate"
	@echo "  macOS/Linux: source venv/bin/activate"
	@echo "Then run: make install"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	@echo "Cleaned up generated files!"

train:
	python src/model.py
	@echo "Model training complete!"

predict:
	python src/predict.py
	@echo "Predictions generated!"

submit:
	python -c "from src.utils import validate_submission; validate_submission('submission.csv', 'sample_submission.csv')"
	@echo "Submission validated!"

test:
	pytest tests/ -v
	@echo "Tests complete!"

lint:
	flake8 src/ --max-line-length=120
	@echo "Linting complete!"

format:
	black src/ --line-length=120
	@echo "Code formatted!"

notebook:
	jupyter notebook
	@echo "Jupyter notebook launched!"

# Quick workflow commands
quick-train: clean train
	@echo "Quick training complete!"

quick-submit: predict submit
	@echo "Submission ready!"

full-pipeline: clean train predict submit
	@echo "Full pipeline complete!"

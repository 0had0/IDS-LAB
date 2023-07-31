initialize_git:
	@echo "Initializing git..."
	git init 
	
install: 
	@echo "Installing..."
	poetry install
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	poetry shell

download_data:
	@echo "Downloading data..."
	wget http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip -P data/raw
	unzip data/raw/MachineLearningCSV.zip -d data/raw
	mv data/raw/MachineLearningCVE/* data/raw
	rm -rf data/raw/MachineLearningCVE
	rm data/raw/MachineLearningCSV.zip

setup: initialize_git install download_data

data/processed/dataset.csv: data/raw
	@echo "Preprocessing data..."
	python src/preprocess.py

data/final/test.pkl: data/processed/dataset.csv
	@echo "Creating test & train sets..."
	python src/process.py

models/centralized.pkl: data/final/test.pkl
	@echo "Create & Optimize Centralized Model..."
	python src/train_model.py CENTRALIZED

test:
	pytest

docs_view:
	@echo View API documentation... 
	PYTHONPATH=src pdoc src --http localhost:8080

docs_save:
	@echo Save documentation to docs... 
	PYTHONPATH=src pdoc src -o docs

data/processed/$(model_id).pkl: data/raw src/process.py
	@echo "Processing data..."
	python src/process.py $(model_id)

models/$(model_id).pkl: data/processed/xy.pkl src/train_model.py
	@echo "Training model..."
	python src/train_model.py $(model_id)

notebooks/results.ipynb: models/svc.pkl src/run_notebook.py
	@echo "Running notebook..."
	python src/run_notebook.py

status:
	@echo "Launching $(model_id) pipeline:"

# model_id="CENTRILIZED" | "FEDRATED_GLOBAL" | "<int>TH_LOCAL_CENTRALIZED"
pipeline: status data/processed/dataset.csv data/processed/$(model_id).pkl models/$(model_id).pkl notebooks/results.ipynb

centralized_pipeline_start:
	@echo "Launching Centralized Pipeline..."

centralized_pipeline: centralized_pipeline_start data/final/test.pkl models/centralized.pkl

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
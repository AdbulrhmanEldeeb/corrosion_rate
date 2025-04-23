# Variables
PYTHON=python3
PIP=pip3
MAIN_SCRIPT=src/Corrosion_Rate_Prediction_+_Suggesstions.py
REQ=requirements.txt

.PHONY: install run clean format help

install:
	$(PIP) install -r $(REQ)

run:
	streamlit run $(MAIN_SCRIPT)

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +

format:
	black .

help:
	@echo "Makefile commands:"
	@echo "  install     Install required packages"
	@echo "  run         Run the main Python script"
	@echo "  clean       Remove Python cache files"
	@echo "  format      Format code using Black"
	@echo "  help        Show available commands"

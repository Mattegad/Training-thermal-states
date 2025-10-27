# ========================================================
# Makefile for spectroscopy-reservoir project
# Automates setup, execution, and git operations
# ========================================================

# Default virtual environment name
VENV = .venv

# Python executable inside venv
PYTHON = $(VENV)/bin/python

# Source directory
SRC = src

# ========================================================
# Basic commands
# ========================================================

.PHONY: all setup run train clean push

all: setup run

setup:
	@echo "🔧 Creating virtual environment and installing dependencies..."
	python3 -m venv $(VENV)
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Environment ready."

run: train

train:
	@echo "🚀 Running training / Figure 2 simulation..."
	@$(PYTHON) -m $(SRC).train
	@echo "✅ Simulation complete."

clean:
	@echo "🧹 Cleaning cache and outputs..."
	rm -rf __pycache__/ $(SRC)/*/__pycache__/ output/ .ipynb_checkpoints/
	find . -name '*.pyc' -delete
	@echo "✅ Clean done."

push:
	@echo "📤 Pushing current branch to GitHub..."
	git add .
	git commit -m "Update: automatic push"
	git push
	@echo "✅ Code pushed."

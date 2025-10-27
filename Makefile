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

.PHONY: all setup run train clean push pull

all: setup run

setup:
	@echo "🔧 Setting up virtual environment..."
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install --upgrade --upgrade-strategy eager -r requirements.txt
	@echo "✅ Environment ready."

run: train

train:
	@echo "🚀 Running training / Figure 2 simulation..."
ifeq ($(RE),1)
	@$(PYTHON) -m src.train --replot-only
else
	@$(PYTHON) -m src.train
endif
	@echo "✅ Done."


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

pull:
	@echo "📥 Récupération des dernières modifications depuis GitHub..."
	git pull
	@echo "✅ Code mis à jour."

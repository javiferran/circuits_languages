#!/bin/bash

# Set the path to your virtual environment
VENV_PATH="./venv"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

sleep 2s # Waits 2 seconds.

# Run the Python script
python -u ./create_datasets.py

# Deactivate the virtual environment
deactivate
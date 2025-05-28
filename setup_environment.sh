#!/bin/bash

# Set environment name
ENV_NAME="TreeMorphPipeline"

# Activate conda shell functions
eval "$(conda shell.bash hook)"

# Ensure mamba is installed
if ! command -v mamba &> /dev/null; then
    echo "Mamba not found. Installing it into base..."
    conda install -n base -c conda-forge mamba -y
fi

# Create the environment using mamba if it doesn't already exist
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating environment '$ENV_NAME' with Mamba..."
    mamba env create -f environment.yml
fi

# Activate the environment
echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Install pip requirements
echo "Installing pip packages from requirements.txt..."
pip install -r requirements.txt

# Install local Modules/ package in editable mode
if [ -f "Modules/setup.py" ]; then
    echo "Installing local 'Modules' package in editable mode..."
    pip install -e Modules/
else
    echo "Warning: 'Modules/setup.py' not found — skipping editable install."
fi

echo "Setup complete. Environment '$ENV_NAME' is ready to use."
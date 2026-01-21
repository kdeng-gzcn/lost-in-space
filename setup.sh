#!/bin/bash

# Create a Conda environment
echo "Creating Conda environment..."
conda create -n sr python=3.10 -y

# Activate the Conda environment
echo "Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sr

# Install required dependencies
echo "Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

echo "Conda environment setup complete."
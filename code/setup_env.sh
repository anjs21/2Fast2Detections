#!/bin/bash
# Run this script on the LOGIN NODE (internet access required) before submitting jobs.
# Creates the virtual environment at ~/.env and installs all dependencies.
#
# Usage:
#   bash setup_env.sh

set -euo pipefail

ENV_DIR="$HOME/2Fast2Detections/.env"

# Load the same Python version used by the compute nodes
module load python/3.11.7

# Create venv if it doesn't already exist
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment at $ENV_DIR ..."
    python -m venv "$ENV_DIR"
else
    echo "Virtual environment already exists at $ENV_DIR — skipping creation."
fi

source "$ENV_DIR/bin/activate"

pip install --upgrade pip

# Install PyTorch with CUDA 12.1 (matches Leonardo A100 driver stack)
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install the remaining dependencies
pip install -r "$(dirname "$0")/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu121

pip install clip open_clip_torch

echo ""
echo "Setup complete. Virtual environment is ready at $ENV_DIR"
echo "Activate it with:  source $ENV_DIR/bin/activate"
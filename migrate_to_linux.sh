#!/bin/bash

# Configuration
PROJECT_NAME="inventory_forecasting"
TARGET_DIR="$HOME/projects/$PROJECT_NAME"
SOURCE_DIR=$(pwd)

echo "🚀 Starting Migration from Windows ($SOURCE_DIR) to Linux Native ($TARGET_DIR)..."

# 1. Update and Install Prerequisites
echo "📦 Installing System Dependencies (Python, venv)..."
sudo apt update
sudo apt install -y python3-full python3-pip python3-venv

# 2. Create Target Directory
echo "📂 Creating target directory..."
mkdir -p "$TARGET_DIR"

# 3. Copy Files (Excluding venvs and cache)
echo "COPYING files... (This might take a moment)"
rsync -av --progress "$SOURCE_DIR/" "$TARGET_DIR/" \
    --exclude 'venv' \
    --exclude 'venv_wsl' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store'

echo "✅ Copy complete."

# 4. Setup Virtual Environment in New Location
cd "$TARGET_DIR"
echo "📍 Changed directory to: $(pwd)"

echo "🐍 Creating new Virtual Environment (venv)..."
rm -rf venv # Clean up if exists
python3 -m venv venv

# 5. Activate and Install Requirements
echo "🔌 Activating venv and Installing Requirements..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies with GPU support for TensorFlow
echo "⬇️ Installing libraries..."
pip install -r requirements.txt

# Extra check for GPU support
pip install "tensorflow[and-cuda]"

echo ""
echo "🎉 MIGRATION SUCCESSFUL! 🎉"
echo "To work on your project, run these commands:"
echo "  cd $TARGET_DIR"
echo "  source venv/bin/activate"
echo "  python3 run_batch_experiment.py"

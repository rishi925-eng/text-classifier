#!/usr/bin/env bash
# Build script for Render

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Train the model
echo "Training the model..."
python train_model.py

echo "Build completed successfully!"
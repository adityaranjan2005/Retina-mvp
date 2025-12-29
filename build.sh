#!/bin/bash
set -e

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Starting model download..."
python download_model.py

echo "==> Build complete!"

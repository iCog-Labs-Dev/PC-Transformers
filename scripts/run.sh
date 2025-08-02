#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Step 1: Setting up environment and installing dependencies..."
bash "$SCRIPT_DIR/setup.sh"

echo "Step 2: Downloading and preparing dataset..."
bash "$SCRIPT_DIR/download_data.sh"

echo "Step 3: Running GPT-2 tokenizer..."
source "$SCRIPT_DIR/../venv/bin/activate"
python -m Data_preprocessing.tokenizer.gpt2_tokenizer

echo ""
echo "All steps complete!"

#!/bin/bash

FILE_ID="1yB8f1B-VVXdGRPWf2aYintDoMOmgXYRN"
ZIP_NAME="opwb.zip"
OUTPUT_DIR="Data_preprocessing/Data"

if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install gdown
fi

mkdir -p "$OUTPUT_DIR"
echo "Downloading dataset..."
gdown --id "$FILE_ID" -O "$OUTPUT_DIR/$ZIP_NAME"

echo "Extracting files to $OUTPUT_DIR..."
unzip -o "$OUTPUT_DIR/$ZIP_NAME" -d "$OUTPUT_DIR"

rm "$OUTPUT_DIR/$ZIP_NAME"
echo "Dataset ready at $OUTPUT_DIR/opwb/"

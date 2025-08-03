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
python3 -c "import zipfile; zipfile.ZipFile('$OUTPUT_DIR/$ZIP_NAME').extractall('$OUTPUT_DIR')"

if [ -d "$OUTPUT_DIR/opwb/opwb" ]; then
    echo "Flattening nested folder structure..."
    mv "$OUTPUT_DIR/opwb/opwb/"* "$OUTPUT_DIR/opwb/"
    rmdir "$OUTPUT_DIR/opwb/opwb"
fi

echo "Cleaning up zip..."
rm "$OUTPUT_DIR/$ZIP_NAME"
echo "Dataset ready at $OUTPUT_DIR/opwb/"

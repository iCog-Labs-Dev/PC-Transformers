#!/bin/bash

set -e

echo "Using current directory: $(pwd)"
echo "Creating Python virtual environment..."

python3 -m venv venv
source venv/bin/activate

echo "Ensuring requirements.txt exists..."

if [ ! -f "requirements.txt" ]; then
   echo "requirements.txt not found. Creating one..."

   cat <<EOF > requirements.txt
--extra-index-url https://download.pytorch.org/whl/cu121
filelock==3.18.0
fsspec==2025.3.2
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.4.2
sympy==1.13.1
tqdm==4.67.1
typing_extensions==4.13.1
tokenizers==0.21.1
matplotlib==3.10.3
nltk==3.9.1
bert-score==0.3.13
optuna
psutil==7.0.0
torch==2.5.0
EOF
fi

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
read -p "Do you want to install Flash Attention? (y/n): " install_flash

if [[ "$install_flash" == "y" || "$install_flash" == "Y" ]]; then
   echo "Checking for CUDA 12.4 Toolkit..."
   if ! nvcc --version | grep "release 12.4" > /dev/null; then
       echo "CUDA 12.4 not detected. Installing..."
       sudo apt install -y cuda-toolkit-12-4
   fi

   echo "Downloading and installing Flash Attention..."
   wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   pip install flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
else
   echo "Skipping Flash Attention installation."
fi

echo ""
echo "Adding CUDA 12.4 to environment..."
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo ""
echo "Installation complete!"
echo "Activate your venv with: source venv/bin/activate"


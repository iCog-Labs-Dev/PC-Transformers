# PC-Transformers

## Overview

PC-Transformers explore a new paradigm in transformer models by integrating Predictive Coding principles, shifting away from global backpropagation and towards local, biologically-inspired, prediction-based learning. In this framework, each layer predicts the next layer’s activity and updates itself to minimize its own prediction error, mirroring how the brain may process information.

This repository implements PC-Transformers and evaluates them on classic language modeling tasks, using the [Penn Treebank dataset](https://www.kaggle.com/datasets/aliakay8/penn-treebank-dataset/data), and a subset of OpenWebText.

---

## Installation

```bash
git clone https://github.com/iCog-Labs-Dev/PC-Transformers.git
cd PC-Transformers
python3 -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```
### Using FlashAttention

FlashAttention works well with the following settings(Further Updates Pending):

| Component           | Required Version                  |
|---------------------|-----------------------------------|
| **Ubuntu**          | 22.04                             |
| **Python**          | 3.10.x                            |
| **CUDA Toolkit**    | 12.4                              |
| **NVIDIA Driver**   | ≥ **545.23** (for CUDA 12.3+)     |
| **PyTorch**         | 2.5.0 (`cxx11abi=FALSE`)          |
| **GPU Architecture**| Ampere / Ada / Hopper (e.g., A100, 3090, 4090, H100) |


Verify CUDA and NVIDIA driver:

```bash
nvcc --version
nvidia-smi
```
If not please install the advised Nvidia Driver and Cuda + Cuda Toolkits:

```bash
apt install cuda-toolkit-12-4
```

Download and Install Flash Attention:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
```

Optional: Set environment variables (if not already configured)

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```
---

## Usage
**Prepare Dataset**

▪️ If using Penn Treebank (PTB): No additional steps are needed. The dataset is already included in the repository.

▪️ If using OpenWebText Subset (OPWB): Download it using the following script:
```bash
bash scripts/download_data.sh

```
**Tokenize Data**

Make sure the correct dataset is specified in Config.DATASET_NAME:
```
# inside Data_preprocessing/config.py
DATASET_NAME = "ptb"   # for Penn Treebank
# or
DATASET_NAME = "opwb"  # for OpenWebText subset
```
Then:
```bash
python -m Data_preprocessing.tokenizer.gpt2_tokenizer

```

**Train the Model**
```bash
torchrun --nproc-per-node=<NUM_GPUS> training.py
```
**Evaluate the Model**
```bash
torchrun --nproc-per-node=<NUM_GPUS> eval.py
```
**Generate Text**
```bash
torchrun --nproc-per-node=<NUM_GPUS> generate_text.py
```

---

### Optional Runtime Flags
| Flag            | Description                                | Default                |
| --------------- | ------------------------------------------ | ---------------------- |
| `--flash`       | Enables FlashAttention                     | `False`                |
| `--prompt`      | Input prompt for generation                | `None` (from test set) |
| `--max_tokens`  | Maximum number of tokens to generate       | `50`                   |
| `--temperature` | Sampling temperature (controls randomness) | `1.0`                  |

Example with FlashAttention:

```bash
torchrun --nproc-per-node=<NUM_GPUS> generate_text.py --flash --prompt "once upon a time" --max_tokens 20 --temperature 3.4
```

## Model Structure

- **Embedding Layer:** Maps tokens and positions to vectors.
- **Attention Block:** Predicts contextualized representations using local error and diversity regularization.
- **MLP Block:** Two-layer feedforward, each minimizing local prediction error.
- **Output Layer:** Projects final representation to vocabulary logits for next-token prediction.

Each layer iteratively refines its latent state and weights over T steps using only local signals.

---

## Contact

For questions or contributions, please open an issue.

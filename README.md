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

## Usage
**Tokenize Data:**
```bash
python -m Data_preprocessing.tokenizer.bpe_tokenizer
```
**Train Model:**
```bash
torchrun --nproc-per-node=<NUM_GPUS> training.py
```
**Evaluate:**
```bash
torchrun --nproc-per-node=<NUM_GPUS> eval.py
```
**Generating Text:**
```bash
torchrun --nproc-per-node=<NUM_GPUS> generate_text.py
```

---

## Model Structure

- **Embedding Layer:** Maps tokens and positions to vectors.
- **Attention Block:** Predicts contextualized representations using local error and diversity regularization.
- **MLP Block:** Two-layer feedforward, each minimizing local prediction error.
- **Output Layer:** Projects final representation to vocabulary logits for next-token prediction.

Each layer iteratively refines its latent state and weights over T steps using only local signals.

---

## Contact

For questions or contributions, please open an issue 

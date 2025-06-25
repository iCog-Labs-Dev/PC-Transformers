# PC-Transformers: A Biologically Plausible Transformer Model with Predictive Coding

## Overview

The PC-Transformers model combines the Transformer architecture with Predictive Coding to enable layer-wise prediction and local weight updates. Each layer predicts the next layer's activity, computes the prediction error, infers latent states, and updates weights before passing the prediction forward. This approach aims to mimic biologically plausible learning mechanisms while maintaining the performance benefits of Transformers. For training and inference, the project leverages a 200-million-word subset of the OpenWebText dataset, a large-scale web-based text corpus. The data is splitted into train, validation, and test sets. The model is implemented in PyTorch and uses the Hugging Face Transformers library for the Transformer architecture.

## Model Architecture

<img src="assets/Model_diagram.png" alt="Model Diagram" height="500" width="400" />

### 1. PC Layer

The PCLayer implements the predictive coding (PC) mechanism at each layer of the transformer. It serves as a local learning and inference engine that:

- Infers latent activities (x) that minimize the error between predicted output (μ) and target activity.
- Locally updates weights using Hebbian learning, based on prediction errors.

### 2. Transformer Components

The PCTransformer model is composed of the following components, all integrated with the PCLayer for predictive coding-based inference:

- **Embedding Layer**
  - Predicts initial transformer block input (x_attn1) from word and position embeddings.

- **Transformer Blocks**
  - Each TransformerBlock includes:
    - Attention: Uses predictive coding for both attention output and internal score projection via Q, K, V projections.
    - MLP: feed forward layer 1 (fc1) predicts the input to feed forward layer (fc2), fc2 predicts next layer activity.

- **Output Layer**
  - Final Linear layer that predicts token logits.
  - Uses PCLayer to perform local updates and iterative inference.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/iCog-Labs-Dev/PC-Transformers.git
cd PC-Transformers
pip install -r requirements.txt
```

## Handling Large Data with Git LFS

This branch uses **Git Large File Storage (Git LFS)** to manage large dataset files that are too big for normal Git.

### If you are cloning the repo for the first time and want to get the dataset files:

1. Install Git LFS:

```bash
git lfs install
```

2. Pull the repo:

```bash
git pull
```

### If you are cloning the repo for the first time and want to get the dataset files:

1. Install Git LFS:

```bash
git lfs install
```

2. Pull the repo:

```bash
git pull
```

### If you are cloning the repo for the first time and want to get the dataset files:

1. Install Git LFS:

```bash
git lfs install
```

2. Pull the repo:

```bash
git pull
```

## Usage

- Tokenize the data:

```bash
python -m Data_preprocessing.tokenizer.gpt2_tokenizer
```

- Train the model:

```bash
python training.py
```

- Evaluate the model:

```bash
python eval.py
```

# PC-Transformers

## **Overview**

The PC-Transformers model combines the Transformer architecture with Predictive Coding to enable layer-wise prediction and local weight updates. Each layer predicts the next layer's activity, computes the prediction error, infers latent states, and updates weights. This approach aims to mimic biologically plausible learning mechanisms while maintaining the performance benefits of Transformers. 

The project leverages the [Penn Treebank dataset](https://www.kaggle.com/datasets/aliakay8/penn-treebank-dataset/data
) for training and inference.

## **Model Architecture**
<img src="assets/Model_diagram.png" alt="Model Diagram" height="500" width = "400"/>

### 1. PC Layer
The PCLayer implements the predictive coding mechanism at each transformer layer. It functions as a local inference and learning module that infers latent activities by minimizing the prediction error between the predicted output and the target activity. Additionally, it performs local weight updates using Hebbian learning for the layer weights and applies an Anti-Hebbian learning rule to lateral connections, with all updates driven by the layer’s prediction errors.

### 2. Transformer Components
The PCTransformer model consists of several key components, each integrated with the PCLayer to enable predictive coding-based inference. The Embedding Layer predicts the initial input to the transformer blocks by combining word and position embeddings. Each Transformer Block contains two main parts: an Attention mechanism, which applies predictive coding not only to the attention output but also internally to the score projection through the query, key, and value projections; and an MLP, where the first feedforward layer predicts the input to the second feedforward layer, and the second layer predicts the activity of the next transformer block. Finally, the Output Layer is a linear layer that produces token logits and also employs the PCLayer to perform local weight updates and iterative inference.

## Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/iCog-Labs-Dev/PC-Transformers.git
cd PC-Transformers
```
Create and activate a virtual environment (optional but recommended):
```
python3 -m venv venv
source venv/bin/activate 
```
Install required Python packages:
```
pip install -r requirements.txt
```
## Usage:
Tokenize the data:
```bash
python -m Data_preprocessing.tokenizer.bpe_tokenizer 
```
Train the model:
```bash
Usage:
    torchrun --nproc-per-node=<NUM_GPUS> training.py --flash
    python training.py [--flash]

Flags:
    --flash              Enable FlashAttention for attention layers (default: False)

```
Evaluate the model:
```bash
Usage:
    torchrun --nproc-per-node=<NUM_GPUS> eval.py --flash
    python eval.py [--flash]

Flags:
    --flash              Enable FlashAttention for attention layers (default: False)
```
Generating text:
```bash
Usage:
    torchrun --nproc-per-node=<NUM_GPUS> generate_text.py --flash --max_tokens 100 --prompt "Once upon a time"
    python generate_text.py [--flash] [--max_tokens N] [--prompt PROMPT] [--temperature T]

Flags:
    --flash              Enable FlashAttention for attention layers (default: False)
    --max_tokens N       Maximum number of new tokens to generate (default: 50)
    --prompt PROMPT      Text prompt to use for generation (default: None, uses test set)
    --temperature T      Sampling temperature (default: 1.0)
```

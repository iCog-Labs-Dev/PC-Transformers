# PC-Transformers

A Transformer model implementation integrated with Predictive Coding (PC) principles, where each layer predicts the next layer's activity, computes errors, and updates weights locally. This project leverages the Penn Treebank dataset for training and inference.

---

## **Overview**

The **PC-Transformers** model combines the Transformer architecture with Predictive Coding to enable layer-wise prediction and local weight updates. Each layer predicts the next layer's activity, computes the prediction error, infers latent states, and updates weights before passing the prediction forward. This approach aims to mimic biologically plausible learning mechanisms while maintaining the performance benefits of Transformers.

### **Key Features**
- **Predictive Coding (PC) Layer**: Implements iterative inference and local Hebbian-style weight updates.
- **Top-Down Prediction Chain**: 
  - Embedding → Attention → MLP → Output.
  - Each layer predicts the next layer's activity.
- **Local Learning**: Layers update their weights autonomously using prediction errors.
- **Modular Design**: Components include embeddings, multi-head attention, MLP blocks, and transformer blocks.

---

## **Model Architecture**

### **Core Components**

#### 1. **`PCLayer` (Predictive Coding Layer)**
- Implements predictive coding logic with iterative latent state updates.
- Handles error computation, weight updates, and activity clamping.
- Supports embeddings, attention, MLP, and output layers.
- Key features:
  - **Iterative Inference**: Performs `T` steps of latent state refinement.
  - **Local Weight Updates**: Uses Hebbian-style updates with a configurable learning rate.
  - **Error Clamping**: Prevents exploding activations via `clamp_value`.

#### 2. **Embedding Layer**
- Combines token and positional embeddings.
- Uses `PCLayer` to predict downstream attention layer activities.
- Subcomponents:
  - **Token Embeddings**: `nn.Embedding` layer for token IDs.
  - **Positional Embeddings**: `nn.Embedding` layer for positional IDs.
  - **LayerNorm**: Normalizes combined embeddings.

#### 3. **Attention Layer**
- Multi-head self-attention mechanism.
- Predicts MLP layer activities via `PCLayer`.
- Subcomponents:
  - **Query/Key/Value Projections**: Linear layers (`nn.Linear`) for attention computations.
  - **Attention Output**: Linear projection layer (`nn.Linear`).
  - **Softmax + Dropout**: For attention weight stabilization.

#### 4. **MLP Layer**
- Two-layer feed-forward network with GELU activation.
- Predicts output layer activities.
- Structure:
  - **FC1**: Expands dimension to `4 * n_embed` (GELU-activated).
  - **FC2**: Contracts back to `n_embed` dimension.
  - **Dropout**: Applied after FC2.

#### 5. **Transformer Block**
- Sequential stack of attention and MLP layers.
- Workflow:
  1. **Attention Block**:
     - LayerNorm → Multi-head Attention → Residual connection.
  2. **MLP Block**:
     - LayerNorm → MLP → Residual connection.
- Uses `PCLayer` for inter-layer prediction and error propagation.

#### 6. **Output Layer**
- Final linear projection to vocabulary size.
- Predicts token logits using `PCLayer`.
- Structure:
  - **Linear Layer**: Maps hidden states to logits (`nn.Linear`).
  - **Error Propagation**: Connects to the final loss computation.


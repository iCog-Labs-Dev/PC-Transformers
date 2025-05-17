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


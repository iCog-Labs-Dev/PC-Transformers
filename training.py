import torch
import time
import torch.nn.functional as F
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from model_architecture.pc_t_model import PCTransformer
from Data_preprocessing.dataloader import train_loader
from utils.model_utils import load_tokenizer, reset_pc_modules
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

"""
Usage: python training.py

This script trains a predictive coding transformer model on a dataset.
It tracks and plots the average predictive coding energy per epoch and saves the trained model.
"""

def train(model, dataloader):
    """
    Train the model for one epoch on the provided dataloader.

    Args:
        model: The model to train.
        dataloader: DataLoader providing batches of input and target tensors.

    Returns:
        avg_energy (float): Average predictive coding energy or CE loss per batch for the epoch.
    """
    model.train()
    total_energy = 0.0
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]

        logits = model(target_ids, input_ids)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=0
        )

        layer_energies = []
        for module in model.modules():
            if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is not None:
                    layer_energies.append(energy)

        # Compute average energy for current batch
        batch_energy = ce_loss.item() if not layer_energies else sum(layer_energies) / len(layer_energies)
        total_energy += batch_energy
        batch_count += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Batch Energy: {batch_energy:.4f}", flush=True)

        reset_pc_modules(model)

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    return avg_energy

tokenizer = load_tokenizer()
vocab_size = tokenizer.get_vocab_size()

config = GPTConfig(
    vocab_size = vocab_size,
    block_size= 256,
    n_embed=64,
    dropout=0.1,
    local_learning_rate=1e-5,
    T=5,
    is_holding_error = True,
    num_heads=2,
    n_blocks=4,
    num_epochs=5,
    update_bias=True,
    use_lateral = True,
    energy_fn_name="scaled_mse" 
)

model = PCTransformer(config)
train_energies = []

print("========== Training started ==========", flush=True) 
# Measure total training time
start_training_time = time.time()
for epoch in range(config.num_epochs):
    print(f"Epoch {epoch+1} started", flush=True)
    avg_energy = train(model, train_loader)
    train_energies.append(avg_energy)
    print(f"Epoch {epoch+1} | Avg Energy: {avg_energy:.4f}", flush=True)
total_training_time = time.time() - start_training_time
print(f"Total Training Time: {total_training_time:.2f} seconds", flush=True)
print("========== Training completed ==========", flush=True)

# Saving trained model
torch.save({"model_state": model.state_dict()}, "checkpoints/pc_transformer.pt")
print("Model saved.")

# Plotting average energy vs. epoch
epochs = list(range(1, len(train_energies) + 1))
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_energies, marker='o', linestyle='-', color='b', label='Average Batch Energy')
plt.xlabel('Epoch')
plt.ylabel('Average Batch Energy')
plt.title('Average Batch Energy vs. Epoch')
plt.grid(True)
plt.legend()
# Force x-axis to show only whole numbers
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('assets/energy_plot.png')
plt.show()

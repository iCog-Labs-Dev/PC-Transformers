import os
import torch
import os
import math
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

def train(model, dataloader, tokenizer):
    model.train()
    total_energy = 0.0
    total_ce_loss = 0.0
    batch_count = 0
    pad_token_id = tokenizer.token_to_id("[PAD]")

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]

        logits = model(target_ids, input_ids)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index= pad_token_id
        )
        
        total_ce_loss += ce_loss.item()

        layer_energies = []
        for module in model.modules():
            if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is not None:
                    layer_energies.append(energy)
                if hasattr(module, "_head_similarity"):
                    _ = module._head_similarity_avg
                    _ = module._head_similarity_max

        # Compute average energy for current batch
        batch_energy = ce_loss.item() if not layer_energies else sum(layer_energies) / len(layer_energies)
        total_energy += batch_energy
        batch_count += 1
        perplexity = math.exp(ce_loss.item()) if ce_loss.item() < 100 else float("inf")
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Batch Energy: {batch_energy:.4f} | Perplexity: {perplexity:.4f}", flush=True)

        reset_pc_modules(model)

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")    
    return avg_energy, avg_perplexity

def main():
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    config = GPTConfig(
        vocab_size = vocab_size,
        block_size= 256, 
        n_embed=64,
        dropout=0.1,
        local_learning_rate=1e-5,
        T= 20,
        is_holding_error = True,
        num_heads=5,
        n_blocks=4,
        num_epochs= 50,
        update_bias=True,
        use_lateral = True,
        energy_fn_name="scaled_mse",
        eos_token_id = tokenizer.token_to_id("[EOS]")
    )
    model = PCTransformer(config)
    train_energies = []
    perplexities = []

    print("========== Training started ==========", flush=True) 
    # Measure total training time
    start_training_time = time.time()
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1} started", flush=True)
        avg_energy, perplexity = train(model, train_loader, tokenizer)
        train_energies.append(avg_energy)
        perplexities.append(perplexity)
        print(f"Epoch {epoch+1} | Avg Energy: {avg_energy:.4f} | Perplexity: {perplexity:.4f}", flush=True)
    total_training_time = time.time() - start_training_time
    print(f"Total Training Time: {total_training_time:.2f} seconds", flush=True)
    print("========== Training completed ==========", flush=True)

    # Saving trained model
    save_path = "checkpoints/pc_transformer.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        os.remove(save_path)
    torch.save({"model_state": model.state_dict()}, save_path)
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

if __name__ == "__main__":
    main()

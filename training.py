import torch
import os
import torch.nn.functional as F
from tokenizers import Tokenizer
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from model_architecture.pc_t_model import PCTransformer
from Data_preprocessing.dataloader import train_loader
from Data_preprocessing.config import Config

"""Usage: python training.py"""

def train(model, dataloader):
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

        for module in model.modules():
            if hasattr(module, "clear_energy"):
                module.clear_energy()
            if hasattr(module, "clear_errors"):
                module.clear_errors()

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    return avg_energy

tokenizer_path = os.path.join(Config.TOKENIZER_DIR, "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

config = GPTConfig(
    vocab_size = vocab_size,
    block_size= 256,
    n_embed=64,
    dropout=0.1,
    local_learning_rate=1e-7,
    T=1,
    is_holding_error = True,
    num_heads=2,
    n_blocks=2,
    num_epochs=5,
    update_bias=True,
)

model = PCTransformer(config)
train_energies = []

print("========== Training started ==========", flush=True) 
for epoch in range(config.num_epochs):
    print(f"Epoch {epoch+1} started", flush=True)
    avg_energy = train(model, train_loader)
    train_energies.append(avg_energy)
    print(f"Epoch {epoch+1} | Avg Energy: {avg_energy:.4f}", flush=True)

# Save trained model
save_path = "checkpoints/pc_transformer.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

if os.path.exists(save_path):
    os.remove(save_path)
    
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
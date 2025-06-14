import os
import time
import torch
from tokenizers import Tokenizer
from model_architecture.pc_t_model import PCTransformer
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from Data_preprocessing.dataloader import test_loader
from Data_preprocessing.config import Config
import torch.nn.functional as F

def evaluate(model, dataloader):
    model.eval()
    total_energy = 0.0
    batch_count = 0
    total_ce_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"]
            targets = batch["target_ids"]

            logits = model(targets, input_ids)

            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )
            total_ce_loss += ce_loss.item()
            
            layer_energies = []
            for module in model.modules():
                if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                    energy = module.get_energy()
                    if energy is not None:
                        layer_energies.append(energy)
            
            batch_energy = ce_loss.item() if not layer_energies else sum(layer_energies) / len(layer_energies)
            total_energy += batch_energy
            batch_count += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} | CE Loss: {ce_loss.item():.4f}| Batch Energy: {batch_energy:.4f}", flush=True)

            # Clear energies and errors for next batch
            for module in model.modules():
                if hasattr(module, "clear_energy"):
                    module.clear_energy()
                if hasattr(module, "clear_errors"):
                    module.clear_errors()

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    return avg_energy, avg_ce_loss 

def generate_text(model, input_ids, config, max_new_tokens=50, temperature=1.0):
    model.eval()
    input_tensor = input_ids.unsqueeze(0)

    for _ in range(max_new_tokens):
        if input_tensor.size(1) > config.block_size:
            input_tensor = input_tensor[:, -config.block_size:]
        
        with torch.no_grad():
            logits = model(input_tensor, input_tensor)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tensor = torch.cat((input_tensor, next_token), dim=1)
        
        for module in model.modules():
            if hasattr(module, "clear_errors"):
                module.clear_errors()
            if hasattr(module, "clear_energy"):
                module.clear_energy()

    return input_tensor[0]

tokenizer_path = os.path.join(Config.TOKENIZER_DIR, "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

config = GPTConfig(
    vocab_size = vocab_size,
    block_size=256,
    n_embed=64,
    dropout=0.1,
    local_learning_rate=1e-5,
    T=2,
    is_holding_error=True,
    num_heads=2,
    n_blocks=4,
    num_epochs=1,
    update_bias=True,
    energy_fn_name="kld"
)

model = PCTransformer(config)
checkpoint = torch.load("checkpoints/pc_transformer.pt")
model.register_all_lateral_weights() 
model.load_state_dict(checkpoint["model_state"])

print("========== Evaluation started ==========", flush=True)
start_time = time.time()
avg_energy, avg_ce_loss = evaluate(model, test_loader) 
elapsed = time.time() - start_time

print(f"Average Energy on Evaluation Set: {avg_energy:.4f}")
print(f"Average CE Loss on Evaluation Set: {avg_ce_loss:.4f}")
print(f"Evaluation Time: {elapsed:.2f} seconds")
print("========== Evaluation completed ==========", flush=True)


# Generate text using the trained model
for batch in test_loader:
    input_ids = batch["input_ids"][0]
    target_ids = batch["target_ids"][0]
    break

prompt_length = 10
prompt_ids = input_ids[:prompt_length]
generated_ids = generate_text(model, prompt_ids, config, max_new_tokens=50, temperature=0.7)

print("\n Prompt:")
print(tokenizer.decode(prompt_ids.tolist()))

print("\n Target")
print(tokenizer.decode(input_ids.tolist()))

print("\n Generated Text:")
print(tokenizer.decode(generated_ids.tolist()))

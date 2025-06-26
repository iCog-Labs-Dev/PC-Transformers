import time
import torch
import math
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from Data_preprocessing.dataloader import test_loader
import torch.nn.functional as F
from utils.model_utils import load_tokenizer, load_model, reset_pc_modules

"""Usage: python eval.py"""

def evaluate(model, dataloader, tokenizer, max_batches=None):
    start_time = time.time()
    model.eval()
    
    total_energy = 0.0
    batch_count = 0
    total_ce_loss = 0.0
    pad_token_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size
    
    if max_batches is None:
        print(f"Evaluating on the full test set...")
    else:
        print(f"Evaluating on up to {max_batches} batches...")
        
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"]
        targets = batch["target_ids"]

        if (targets == pad_token_id).sum() == 0:
            print(f"No pad tokens detected in batch {batch_idx + 1}, check padding behavior.")

        # Clip targets to valid range before using them for loss calculation
        if targets.max() >= vocab_size:
            targets = torch.clamp(targets, max=vocab_size-1)

        logits = model(targets, input_ids)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=pad_token_id,
        )
        total_ce_loss += ce_loss.item()

        layer_energies = []
        for module in model.modules():
            if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is not None and not torch.isnan(torch.tensor(energy)):
                    layer_energies.append(energy)

        if layer_energies:
            valid_energies = [e for e in layer_energies if not torch.isnan(torch.tensor(e))]
            batch_energy = sum(valid_energies) / len(valid_energies) if valid_energies else ce_loss.item()
        else:
            batch_energy = ce_loss.item()

        total_energy += batch_energy
        batch_count += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | CE Loss: {ce_loss.item():.4f}| Batch Energy: {batch_energy:.4f}", flush=True)

        reset_pc_modules(model)

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")

    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.2f} seconds")
    print(f"Total Batches Processed: {batch_idx + 1}")
    print(f"Avg CE Loss: {avg_ce_loss:.4f} | Avg Energy: {avg_energy:.4f} | Avg Perplexity: {avg_perplexity:.4f}")

    return avg_energy, avg_ce_loss, avg_perplexity

def main():
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.vocab_size
    config = GPTConfig(
        vocab_size = vocab_size,
        block_size=208,
        n_embed= 208,
        dropout= 0.07813827928828256,
        local_learning_rate= 1.51e-05,
        T=20,
        is_holding_error=True,
        num_heads= 16,
        n_blocks=6,
        num_epochs=1,
        update_bias=True,
        energy_fn_name="scaled_mse", 
        eos_token_id = tokenizer.eos_token_id
    )

    model_path = "checkpoints/pc_transformer.pt"
    model = load_model(model_path, config)

    # Max batches can be set to limit evaluation, or None for full dataset
    evaluate(model, test_loader, tokenizer, max_batches= None)

if __name__ == "__main__":
    main()

import time
import math
import torch
import os
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from Data_preprocessing.dataloader import get_loaders
import torch.nn.functional as F
from utils.model_utils import load_tokenizer, load_model, reset_pc_modules
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

"""Usage: python eval.py"""
local_rank = int(os.getenv("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader, tokenizer, max_batches=None, device=None):        
    model = model.to(device)

    start_time = time.time()
    model.eval()
    total_energy = 0.0
    batch_count = 0
    total_ce_loss = 0.0
    pad_token_id = tokenizer.token_to_id("[PAD]")
    
    if max_batches is None:
        print(f"Evaluating on the full test set...")
    else:
        print(f"Evaluating on up to {max_batches} batches...")
        
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        input_ids = batch["input_ids"].to(device)
        targets = batch["target_ids"].to(device)

        logits = model(targets, input_ids)
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index= pad_token_id,
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

        reset_pc_modules(model)

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")
 
    elapsed = (time.time() - start_time) / 3600
    if local_rank == 0:
        print(f"Evaluation completed in {elapsed:.2f} hours")
        print(f"Total Batches Processed: {batch_idx}")
        print(f"Avg CE Loss: {avg_ce_loss:.4f} | Avg Energy: {avg_energy:.4f}")

    return avg_energy, avg_ce_loss, avg_perplexity

def main():
    dist.init_process_group(backend="nccl")
    print(f"[Rank {local_rank}] Using device: {device}")

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    config = GPTConfig(
        vocab_size = vocab_size,
        block_size=256,
        n_embed=64,
        dropout=0.1,
        local_learning_rate= 1e-5,
        T=20,
        is_holding_error=True,
        num_heads=8,
        n_blocks=4,
        num_epochs=1,
        update_bias=True,
        energy_fn_name="scaled_mse", 
        eos_token_id = tokenizer.token_to_id("[EOS]")
    )

    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path, config)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    _, _, test_loader = get_loaders(distributed=True)

    # Max batches can be set to limit evaluation, or None for full dataset
    evaluate(model, test_loader, tokenizer, max_batches = None, device = device)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
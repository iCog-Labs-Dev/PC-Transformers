import time
import math
import torch
import math
import os
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from Data_preprocessing.dataloader import get_loaders
import torch.nn.functional as F
from utils.model_utils import load_tokenizer, load_model, reset_pc_modules
from utils.pc_utils import cleanup_memory
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

"""
This script evaluates the performance of the predictive coding transformer model.

Usage: torchrun --nproc-per-node=<NUM_GPU> eval.py

"""

local_rank = int(os.getenv("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

local_rank = int(os.getenv("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader, tokenizer, max_batches=None, device = None):
    model.eval()
    total_energy = 0.0
    batch_count = 0
    total_ce_loss = 0.0
    pad_token_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)
    
    base_model = model.module if hasattr(model, 'module') else model
    output_pc_layer = base_model.output.pc_layer

    alpha = getattr(base_model.config, 'combined_internal_weight', 0.3)
    beta = getattr(base_model.config, 'combined_output_weight', 0.7)
    
    if local_rank == 0:
        if max_batches is None:
            print(f"Evaluating on the full test set...")
        else:
            print(f"Evaluating on up to {max_batches} batches...")
        
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        targets = batch["target_ids"].to(device)

        if local_rank == 0:
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

        internal_energies = []
        output_energy = None

        for module in model.modules():
            if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is None or (isinstance(energy, float) and math.isnan(energy)):
                    continue

                if module is output_pc_layer:
                    output_energy = energy
                else:
                    internal_energies.append(energy)

        avg_internal_energy = sum(internal_energies) / len(internal_energies) if internal_energies else ce_loss.item()
        avg_output_energy = output_energy if output_energy is not None else ce_loss.item()

        batch_energy = alpha * avg_internal_energy + beta * avg_output_energy
        total_energy += batch_energy
        batch_count += 1

        if dist.get_rank() == 0 and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | CE Loss: {ce_loss.item():.4f}| Energy: { batch_energy:.4f}", flush=True)

        reset_pc_modules(model)
        cleanup_memory()

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")

    
    if local_rank == 0:
        print(f"Total Batches Processed: {batch_idx + 1}")
        print(f"Avg CE Loss: {avg_ce_loss:.4f} | Avg Energy: {avg_energy:.4f} | Avg Perplexity: {avg_perplexity:.4f}")

    return avg_energy, avg_perplexity

def main():
    dist.init_process_group(backend="nccl")
    print(f"[Rank {local_rank}] Using device: {device}")

    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)
    config = GPTConfig(
        vocab_size = vocab_size,
        block_size=448,
        n_embed= 592,
        dropout= 0.24684719512514441,
        local_learning_rate= 1e-5,
        T=7,
        is_holding_error=True,
        num_heads= 16,
        n_blocks=6,
        num_epochs=1,
        update_bias=False,
        internal_energy_fn_name="pc_e", 
        output_energy_fn_name="kld",
        eos_token_id = tokenizer.eos_token_id,
        combined_internal_weight=0.3,
        combined_output_weight=0.7,
        use_flash_attention=True
    )

    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path, config)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    _, _, test_loader = get_loaders(distributed = True)

    # Max batches can be set to limit evaluation, or None for full dataset
    start_time = time.time()
    evaluate(model, test_loader, tokenizer, max_batches= None, device=device)
    elapsed = time.time() - start_time
    if local_rank == 0:
        print(f"Evaluation completed in {elapsed:.2f} seconds")
        
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

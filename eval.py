import time
import math
import torch
import math
import os
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from Data_preprocessing.dataloader import get_loaders
from Data_preprocessing.config import Config
import torch.nn.functional as F
from utils.model_utils import load_tokenizer, load_model, reset_pc_modules
from utils.config_utils import load_best_config
from utils.pc_utils import cleanup_memory
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils.device_utils import setup_device, cleanup_memory
import argparse
"""
This script evaluates the performance of the predictive coding transformer model.

Usage: torchrun --nproc-per-node=<NUM_GPU> eval.py

"""

local_rank, device, use_ddp = setup_device()

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

                if hasattr(module, 'layer_type') and module.layer_type == 'linear_output':
                    if getattr(module, 'energy_fn_name', None) == "kld":
                        output_energy = energy
                    else:
                        internal_energies.append(energy)
                else: 
                    internal_energies.append(energy)

        avg_internal_energy = sum(internal_energies) / len(internal_energies) if internal_energies else ce_loss.item()
                
        if output_energy is not None:
           avg_output_energy = output_energy
           batch_energy = alpha * avg_internal_energy + beta* avg_output_energy 
        else:
            batch_energy = avg_internal_energy

        total_energy += batch_energy
        batch_count += 1

        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Batch Energy: {batch_energy:.4f}")
        
        reset_pc_modules(model)
        cleanup_memory()
   
    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")
  
    

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Total Batches Processed: {batch_idx + 1}")
        print(f"Avg CE Loss: {avg_ce_loss:.4f} | Avg Energy: {avg_energy:.4f} | Avg Perplexity: {avg_perplexity:.4f}")
    

    return avg_energy, avg_perplexity

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention for attention layers')
    args = parser.parse_args()

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    print(f"[Rank {local_rank}] Using device: {device}")

    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)
    
    best_config = load_best_config()
    
    config = GPTConfig(
        vocab_size = vocab_size,
        block_size = Config.MAX_LENGTH,
        n_embed = best_config["n_embed"],
        dropout = best_config["dropout"],
        local_learning_rate = best_config["peak_learning_rate"],
        T = best_config["T"],
        is_holding_error = True,
        num_heads = best_config["num_heads"],
        n_blocks = best_config["n_blocks"],
        num_epochs = 1,
        internal_energy_fn_name="pc_e",
        output_energy_fn_name="pc_e",
        eos_token_id = tokenizer.eos_token_id,
        combined_internal_weight=0.3,
        combined_output_weight=0.7,
        update_bias = best_config["update_bias"]        
    )
  
    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path, config)
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    _, _, test_loader = get_loaders(distributed = use_ddp)

    # Max batches can be set to limit evaluation, or None for full dataset
    start_time = time.time()
    evaluate(model, test_loader, tokenizer, max_batches= None, device=device)
    elapsed = time.time() - start_time
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Evaluation completed in {elapsed:.2f} seconds")  
        
    if use_ddp and dist.is_initialized(): 
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

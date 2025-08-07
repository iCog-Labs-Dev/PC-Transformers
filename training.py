import torch
import os
import torch.nn as nn
import math
import time
import torch.nn.functional as F
import torch.distributed as dist
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from model_architecture.pc_t_model import PCTransformer
from Data_preprocessing.dataloader import get_loaders
from utils.model_utils import load_tokenizer, reset_pc_modules
from utils.pc_utils import cleanup_memory
from eval import evaluate
from visualization import plot_metrics
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from utils.device_utils import setup_ddp

"""
training.py

This script trains a Predictive Coding Transformer model using the selected dataset.
It supports distributed training and logs energy and perplexity metrics
for both training and validation phases. 


Usage (CPU):
    python training.py --dataset=<dataset_name>

Usage (Multi-GPU Distributed Training):
    torchrun --nproc-per-node=<NUM_GPUS> training.py --dataset=<dataset_name>
    
ARGUMENTS:
    --dataset: Which dataset to use. Options:
        - ptb  : Penn Treebank
        - opwb : OpenWebText

"""


def train(model, dataloader, tokenizer, config, global_step, device):
    model.train()
    total_ce_loss = 0.0
    total_energy = 0.0
    batch_count = 0
    pad_token_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    base_model = model.module if hasattr(model, 'module') else model
    output_pc_layer = base_model.output.pc_layer

    alpha = getattr(config, 'combined_internal_weight', 0.3)
    beta = getattr(config, 'combined_output_weight', 0.7)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        if target_ids.max() >= vocab_size:
            target_ids = torch.clamp(target_ids, max=vocab_size - 1)

        if global_step < config.warmup_steps:
            lr = config.local_learning_rate + global_step / config.warmup_steps * (
                config.peak_learning_rate - config.local_learning_rate)
        else:
            lr = config.peak_learning_rate

        for module in model.modules():
            if hasattr(module, 'local_lr'):
                module.set_learning_rate(lr)
                
        global_step += 1
        if target_ids.max() >= vocab_size:
            target_ids = torch.clamp(target_ids, max=vocab_size-1)
            
        logits = model(target_ids, input_ids)
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=pad_token_id
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

                if hasattr(module, "_head_similarity_avg"):
                    _ = module._head_similarity_avg
                if hasattr(module, "_head_similarity_max"):
                    _ = module._head_similarity_max

        avg_internal_energy = sum(internal_energies) / len(internal_energies) if internal_energies else ce_loss.item()
        avg_output_energy = output_energy if output_energy is not None else ce_loss.item()

        batch_energy = alpha * avg_internal_energy +beta* avg_output_energy
        total_energy += batch_energy
        batch_count += 1

        perplexity = math.exp(ce_loss.item()) if ce_loss.item() < 100 else float("inf")

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Batch Energy: {batch_energy:.4f} | Perplexity: {perplexity:.4f}", flush=True)

        reset_pc_modules(model)
        cleanup_memory()

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")

    return avg_energy, avg_perplexity, global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='opwb', choices=['ptb', 'opwb'], help='Dataset to use (ptb or opwb)')
    args = parser.parse_args()

    # Set dataset in config
    from Data_preprocessing.config import Config
    Config.DATASET_NAME = args.dataset
    print(f"Using dataset: {Config.DATASET_NAME}")

    local_rank, is_distributed = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (local rank {local_rank})")

    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)

    config = GPTConfig(
        vocab_size = vocab_size,
        block_size= 448, 
        peak_learning_rate= 2e-5,
        warmup_steps= 217,
        n_embed=592,
        dropout= 0.24684719512514441,
        local_learning_rate= 0.0,
        T= 10,
        is_holding_error = True,
        num_heads=16,
        n_blocks=6,
        num_epochs= 20,
        update_bias= True,
        use_lateral = True,
        internal_energy_fn_name="mse",
        output_energy_fn_name="kld",
        eos_token_id=tokenizer.eos_token_id,
        combined_internal_weight=0.3,
        combined_output_weight=0.7,
        use_flash_attention=True  
    )

    model = PCTransformer(config).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, 
                    output_device=local_rank if torch.cuda.is_available() else None, 
                    find_unused_parameters=True)
        model.module.register_all_lateral_weights()
    else:
        model.register_all_lateral_weights()

    train_loader, valid_loader, _ = get_loaders(distributed=is_distributed)
    
    start_time = time.time()
    global_step = 0
    train_energies = []
    val_energies = []
    
    rank = dist.get_rank() if is_distributed and dist.is_initialized() else 0
    if rank == 0:
        print("========== Training started ==========", flush=True)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params / 1e6:.2f} M parameters", flush=True)

    for epoch in range(config.num_epochs):
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        

        if rank == 0:
            print(f"Epoch {epoch + 1}/{config.num_epochs}")

        model.train()
        train_energy, train_perplexity, global_step = train(
            model, train_loader, tokenizer, config, global_step, device
        )
        train_energies.append(train_energy)

        
        model.eval()
        val_energy,_,val_perplexity = evaluate(
            model, valid_loader, tokenizer, max_batches=None, device=device
        )
        val_energies.append(val_energy)

        if rank == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs} | "
            f"Train Energy: {train_energy:.4f} | Train Perplexity: {train_perplexity:.4f} | "
            f"Val Energy: {val_energy:.4f} | Val Perplexity: {val_perplexity:.4f}")
            
            if (epoch + 1) % 5 == 0:
                    os.makedirs("checkpoints", exist_ok=True)
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                        'train_energy': train_energy,
                        'val_energy': val_energy,
                        'train_perplexity': train_perplexity,
                        'val_perplexity': val_perplexity
                    }
                    checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pt'
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

    if rank == 0:
        plot_metrics(train_energies, val_energies)
        os.makedirs("checkpoints", exist_ok=True)
        final_checkpoint = {
            'epoch': config.num_epochs,
            'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
            'train_energy': train_energy,
            'val_energy': val_energy,
            'train_perplexity': train_perplexity,
            'val_perplexity': val_perplexity
        }
        torch.save(final_checkpoint, 'checkpoints/final_model.pt')

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print("Final model saved to: checkpoints/final_model.pt")
        print("========== Training completed ==========")
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
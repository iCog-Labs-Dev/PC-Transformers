import torch
import os
import torch.nn as nn
import math
import time
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from model_architecture.pc_t_model import PCTransformer
from Data_preprocessing.dataloader import get_loaders
from utils.model_utils import load_tokenizer, reset_pc_modules
from utils.device_utils import setup_device, cleanup_memory
from eval import evaluate
from visualization import plot_metrics
import argparse

"""
Training Script for Predictive Coding Transformer

This script trains a predictive coding transformer model

Usage:
    python training.py [--flash]

Flags:
    --flash              Enable FlashAttention for attention layers (default: False)

Example:
    python training.py --flash
    torchrun --nproc-per-node=<NUM_GPUS> training.py --flash
"""

def train(model, dataloader, tokenizer, config, global_step, device):
    """
    Run one epoch of training for the predictive coding transformer model.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        tokenizer: Tokenizer object for decoding and padding.
        config (GPTConfig): Model and training configuration.
        global_step (int): The current global step for learning rate scheduling.
        device (torch.device): Device to run training on.

    Returns:
        tuple: (average energy, average perplexity, updated global_step)
    """
    model.train()
    total_energy = 0.0
    total_ce_loss = 0.0
    batch_count = 0
    pad_token_id = tokenizer.token_to_id('[PAD]')
    vocab_size = tokenizer.get_vocab_size()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        if global_step < config.warmup_steps:
            lr = config.local_learning_rate + global_step / config.warmup_steps * (
                config.peak_learning_rate - config.local_learning_rate)
        else:
            lr = GPTConfig.peak_learning_rate

        for module in model.modules():
            if hasattr(module, 'local_lr'):
                module.local_lr = lr

        global_step += 1
        
        if target_ids.max() >= vocab_size:
            target_ids = torch.clamp(target_ids, max=vocab_size-1)

        logits = model(target_ids, input_ids)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=pad_token_id)
        total_ce_loss += ce_loss.item()

        layer_energies = []
        for module in model.modules():
            if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is not None and not (torch.isnan(torch.tensor(energy)) if isinstance(energy, (int, float)) else False):
                    layer_energies.append(energy)
                if hasattr(module, "_head_similarity"):
                    _ = module._head_similarity_avg
                    _ = module._head_similarity_max
                    
        if layer_energies:
            energies_tensor = torch.tensor(layer_energies, device=device)
            valid_energies = energies_tensor[~torch.isnan(energies_tensor)]
            batch_energy = valid_energies.mean().item() if not valid_energies.numel() == 0 else ce_loss.item()
        else:
            batch_energy = ce_loss.item()
        total_energy += batch_energy
        batch_count += 1
        perplexity = math.exp(ce_loss.item()) if ce_loss.item() < 100 else float("inf")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Batch Energy: {batch_energy:.4f} | Perplexity: {perplexity:.4f}")
            # if device.type == "cuda":
            #     print(f"    [Before Cleanup] Allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")

        reset_pc_modules(model)
        cleanup_memory()

        # if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0 and device.type == "cuda":
        #     print(f"    [After Cleanup] Allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")    
    return avg_energy, avg_perplexity, global_step

def main():
    """
    Main entry point for training the predictive coding transformer model.
    Parses command-line arguments, sets up the model, data, and training loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention for attention layers')
    args = parser.parse_args()

    local_rank, device, use_ddp = setup_device()
    print(f"Using device: {device} (local rank {local_rank})")

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    config = GPTConfig(
        vocab_size = vocab_size,
        block_size= 80, 
        peak_learning_rate= 0.0005567991677869024,
        warmup_steps= 175,
        n_embed=656,
        dropout=0.09984621100041206,
        local_learning_rate= 0.0,
        T= 7,
        is_holding_error = True,
        num_heads=16,
        n_blocks=4,
        num_epochs= 30,
        update_bias=False,
        use_lateral = True,
        energy_fn_name="kld",
        eos_token_id = tokenizer.token_to_id("[EOS]"),
        use_flash_attention=args.flash
    )
    model = PCTransformer(config).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], 
                    output_device=local_rank, 
                    find_unused_parameters=True)

        model.module.register_all_lateral_weights()

    train_loader, valid_loader, _ = get_loaders(distributed=use_ddp)
    
    start_time = time.time()
    global_step = 0
    train_energies = []
    val_energies = []

    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        print("========== Training started ==========") 
        print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    for epoch in range(config.num_epochs):
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs}")

        model.train()
        train_energy, train_perplexity, _ = train(model, train_loader, tokenizer, config, global_step, device)
        train_energies.append(train_energy)
        
        model.eval()
        val_energy, val_ce_loss, val_perplexity = evaluate(model, valid_loader, tokenizer, device=device)
        val_energies.append(val_energy)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs} | "
            f"Train Energy: {train_energy:.4f} | Train Perplexity: {train_perplexity:.4f} | "
            f"Val Energy: {val_energy:.4f} | Val Perplexity: {val_perplexity:.4f}")

            if (epoch + 1) % 5 == 0:
                    os.makedirs("checkpoints", exist_ok=True)
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
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
            'model_state_dict': model.module.state_dict(),
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
    
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()
if __name__ == "__main__":
    main()

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


"""
This script trains the predictive coding transformer model on the provided dataset.
It tracks and plots the average predictive coding energy per epoch and saves the trained model.

Usage: torchrun --nproc-per-node=<NUM_GPU> training.py

"""

def setup_ddp():
    """Initialize DDP process group"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def train(model, dataloader, tokenizer, config, global_step, device):
    model.train()
    total_ce_loss = 0.0
    total_energy = 0.0
    batch_count = 0
    pad_token_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    base_model = model.module if hasattr(model, 'module') else model
    output_pc_layer = base_model.output.pc_layer

    alpha = getattr(config, 'combined_internal_weight', 1)
    beta = getattr(config, 'combined_output_weight', 1)

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
        attn_energy = None

        for module in model.modules():
            if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is None or (isinstance(energy, float) and math.isnan(energy)):
                    continue

                if hasattr(module, 'layer_type') and module.layer_type == 'attn':
                    if getattr(module, 'energy_fn_name', None) == "kld":
                       attn_energy = energy
                    else:
                       internal_energies.append(energy)
                else:
                      internal_energies.append(energy)
                if hasattr(module, "_head_similarity_avg"):
                    _ = module._head_similarity_avg
                if hasattr(module, "_head_similarity_max"):
                    _ = module._head_similarity_max

        tot_internal_energy = 0.5 * sum(internal_energies)  if internal_energies else ce_loss.item()
        if attn_energy is not None:
           avg_attn_energy = attn_energy  
           batch_energy = alpha * avg_internal_energy + beta* avg_attn_energy
        else:
            batch_energy=tot_internal_energy
        if batch_energy == 0.0:
           batch_energy = ce_loss.item()
           
        total_energy += batch_energy
        batch_count += 1

        perplexity = math.exp(ce_loss.item()) if ce_loss.item() < 100 else float("inf")

        if dist.get_rank() == 0 and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Energy: {batch_energy:.4f} | "
                  f"Perplexity: {perplexity:.4f}", flush=True)

        reset_pc_modules(model)
        cleanup_memory()

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")

    return avg_energy, avg_perplexity, global_step


def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
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
        internal_energy_fn_name="pc_e",
        attn_energy_fn_name="kld",
        eos_token_id=tokenizer.eos_token_id,
        combined_internal_weight=0.3,
        combined_output_weight=0.7,
        use_flash_attention=True  
    )

    model = PCTransformer(config).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model.module.register_all_lateral_weights()

    train_loader, valid_loader, _ = get_loaders(distributed=True)

    start_time = time.time()
    global_step = 0
    train_energies = []
    val_energies = []

    rank = dist.get_rank() if dist.is_initialized() else 0
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
        val_energy, val_perplexity = evaluate(
            model, valid_loader, tokenizer, max_batches=None, device=device
        )
        val_energies.append(val_energy)

        if rank == 0:
            print(f"Epoch {epoch + 1}/{config.num_epochs} | "
                  f"Train Energy: {train_energy:.4f} | Train Perplexity: {train_perplexity:.4f} | "
                  f"Val Energy: {val_energy:.4f} | Val Perplexity: {val_perplexity:.4f}")

            if (epoch + 1) % 5 == 0 or epoch == config.num_epochs - 1:
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

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
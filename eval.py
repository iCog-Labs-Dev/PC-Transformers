"""
Evaluation Script for Predictive Coding Transformer

This script evaluates a trained predictive coding transformer model on the test set.

Usage:
    python eval.py [--flash]

Flags:
    --flash              Enable FlashAttention for attention layers (default: False)

Example:
    python eval.py --flash
    torchrun --nproc-per-node=<NUM_GPUS> eval.py --flash
"""
import time
import math
import torch
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from Data_preprocessing.dataloader import get_loaders
import torch.nn.functional as F
from utils.model_utils import load_tokenizer, load_model, reset_pc_modules
from utils.device_utils import setup_device, cleanup_memory
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse

"""Usage: torchrun --nproc-per-node=2 eval.py"""
local_rank, device, use_ddp = setup_device()

def evaluate(model, dataloader, tokenizer, max_batches=None, device=None):
    """
    Evaluate the predictive coding transformer model on a dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation data.
        tokenizer: Tokenizer object for decoding and padding.
        max_batches (int, optional): Maximum number of batches to evaluate. If None, evaluate all.
        device (torch.device, optional): Device to run evaluation on.

    Returns:
        tuple: (average energy, average cross-entropy loss, average perplexity)
    """
    model.eval()
    total_energy = 0.0
    batch_count = 0
    total_ce_loss = 0.0
    pad_token_id = tokenizer.token_to_id("[PAD]")
    
    if not dist.is_initialized() or dist.get_rank() == 0:
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

        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | CE Loss: {ce_loss.item():.4f}| Batch Energy: {batch_energy:.4f}")
            # if device.type == "cuda":
            #     print(f"    [Before Cleanup] Allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB | "
            #         f"Reserved: {torch.cuda.memory_reserved(device) / 1e6:.2f} MB")

            reset_pc_modules(model)
            cleanup_memory()
            
            # if device.type == "cuda":
            #     print(f"    [After Cleanup] Allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB | "
            #         f"Reserved: {torch.cuda.memory_reserved(device) / 1e6:.2f} MB")
        else:
            reset_pc_modules(model)
            cleanup_memory()

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Total Batches Processed: {batch_idx + 1}")
        print(f"Avg CE Loss: {avg_ce_loss:.4f} | Avg Energy: {avg_energy:.4f}")

    return avg_energy, avg_ce_loss, avg_perplexity

def main():
    """
    Main entry point for evaluating the predictive coding transformer model.
    Parses command-line arguments, sets up the model, data, and evaluation loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention for attention layers')
    args = parser.parse_args()

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    print(f"[Rank {local_rank}] Using device: {device}")

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    config = GPTConfig(
        vocab_size = vocab_size,
        block_size=80,
        n_embed=656,
        dropout= 0.09984621100041206,
        local_learning_rate= 0.0005567991677869024,
        T=7,
        is_holding_error=True,
        num_heads=16,
        n_blocks=4,
        num_epochs=1,
        update_bias=False,
        energy_fn_name="kld", 
        eos_token_id = tokenizer.token_to_id("[EOS]"),
        use_flash_attention=args.flash
    )

    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path, config).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    _, _, test_loader = get_loaders(distributed=use_ddp)

    # Max batches can be set to limit evaluation, or None for full dataset
    start_time = time.time()
    evaluate(model, test_loader, tokenizer, max_batches = None, device = device)
    elapsed = time.time() - start_time
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Evaluation completed in {elapsed:.2f} seconds")

    if use_ddp and dist.is_initialized():   
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
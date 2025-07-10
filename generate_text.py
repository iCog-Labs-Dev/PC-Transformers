import torch
import os
from predictive_coding.config import GPTConfig
from utils.model_utils import load_tokenizer, load_model, reset_pc_modules, decode_ids, compute_text_metrics
import torch.nn.functional as F
from Data_preprocessing.dataloader import get_loaders
from utils.device_utils import setup_device
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse

"""
Usage: python generate_text.py

This script generates text using a trained predictive coding transformer model.

Usage:
    python generate_text.py [--flash]

Flags:
    --flash              Enable FlashAttention for attention layers (default: False)

Example:
    python generate_text.py --flash
    torchrun --nproc-per-node=<NUM_GPUS> generate_text.py --flash
"""
local_rank, device, use_ddp = setup_device()

def generate_text(model, config, input_ids, max_new_tokens=50, temperature=1.0, device=None):
    """
    Generate text from a prompt using the trained model.

    Args:
        model (nn.Module): The trained model for text generation.
        config (GPTConfig): Model configuration.
        input_ids (torch.Tensor): Input token IDs as prompt.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature.
        device (torch.device, optional): Device to run generation on.

    Returns:
        torch.Tensor: Generated token IDs including the prompt.
    """
    model.eval()
    input_tensor = input_ids.unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        if input_tensor.size(1) > config.block_size:
            input_tensor = input_tensor[:, -config.block_size:]
      
        logits = model(input_tensor, input_tensor)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tensor = torch.cat((input_tensor, next_token), dim=1)
        
        reset_pc_modules(model)
        if next_token.item() == config.eos_token_id:
            break
                
    return input_tensor[0] 

def text_generation(model, config, device=None):
    """
    Run text generation for a batch of prompts and print results.

    Args:
        model (nn.Module): The trained model for text generation.
        config (GPTConfig): Model configuration.
        device (torch.device, optional): Device to run generation on.

    Returns:
        tuple: (list of generated strings, list of target strings)
    """
    model = model.to(device)
    
    decoded_preds = []
    decoded_targets = []
    num_samples = 5
    prompt_len = 5
    
    _, _, test_loader = get_loaders(distributed=use_ddp)
    tokenizer = load_tokenizer()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    
    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device) 

        for i in range(min(num_samples, input_ids.size(0))): 
            prompt_ids = input_ids[i][:prompt_len]
            generated_ids = generate_text(model, config, prompt_ids, max_new_tokens= 50, temperature=0.7, device = device)

            target_continuation = input_ids[i][prompt_len:]
            target_continuation = target_continuation[target_continuation != pad_token_id].tolist()

            generated_continuation = generated_ids[prompt_len:].tolist()

            # Decode all
            prompt_str = decode_ids(tokenizer, prompt_ids.tolist())
            target_str = decode_ids(tokenizer, target_continuation, stop_at_eos=True)
            generated_str = decode_ids(tokenizer, generated_continuation, stop_at_eos=True)

            decoded_preds.append(generated_str)
            decoded_targets.append(target_str)
            
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"\n[Batch {batch_idx + 1}, Sample {i + 1}]")
                print(f"[PROMPT ]: {prompt_str}")
                print(f"[TARGET ]: {target_str}")
                print(f"[PREDICT]: {generated_str}")
            break
    return decoded_preds, decoded_targets

def main():
    """
    Main entry point for text generation using the predictive coding transformer model.
    Parses command-line arguments, sets up the model, and runs generation.
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
        dropout=0.09984621100041206,
        local_learning_rate=0.0005567991677869024,
        T=7,
        is_holding_error=True,
        num_heads=16,
        n_blocks=4,
        num_epochs=1,
        update_bias=False,
        energy_fn_name="mse",
        eos_token_id = tokenizer.token_to_id("[EOS]"),
        use_flash_attention=args.flash
    )

    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path, config).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if not dist.is_initialized() or dist.get_rank() == 0:
        decoded_preds, decoded_targets = text_generation(model, config, device)
        if decoded_preds and decoded_targets and local_rank == 0:
            compute_text_metrics(decoded_preds, decoded_targets)

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
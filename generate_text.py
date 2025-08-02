import torch
import os
from predictive_coding.config import GPTConfig
from utils.model_utils import load_tokenizer, load_model, reset_pc_modules, decode_ids, compute_text_metrics
import torch.nn.functional as F
from Data_preprocessing.dataloader import get_loaders
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
"""
Usage: python generate_text.py

This script generates text using a trained predictive coding transformer model.
It takes a prompt, generates new tokens, and prints the prompt, target, and generated text.
"""

def get_device_and_rank():
    if torch.cuda.is_available():
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        return device, local_rank
    else:
        return torch.device("cpu"), 0

def generate_text(model, config, input_ids, max_new_tokens=50, temperature=1.0, device = None):
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

def text_generation(model, config, device = None,  max_samples=2):
    decoded_preds, decoded_targets = [], []
    prompt_len = 5
    total_samples = 0

    _, _, test_loader = get_loaders(distributed=True)
    tokenizer = load_tokenizer()
    pad_token_id = tokenizer.pad_token_id

    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device) 
        batch_size = input_ids.size(0)

        for i in range(batch_size):
            if total_samples >= max_samples:
                break

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

            if local_rank == 0:
                print(f"\n[Batch {batch_idx + 1}, Sample {i + 1}]")
                print(f"[PROMPT ]: {prompt_str}")
                print(f"[TARGET ]: {target_str}")
                print(f"[PREDICT]: {generated_str}")
            
            total_samples += 1

        if total_samples >= max_samples:
            break

    return decoded_preds, decoded_targets

def main():
    device, local_rank = get_device_and_rank()
    use_ddp = torch.cuda.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if use_ddp:
        dist.init_process_group(backend="nccl")
        print(f"[Rank {local_rank}] Using device: {device} (DDP enabled)")
    else:
        print(f"[CPU/SINGLE GPU] Using device: {device} (DDP disabled)")

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
        energy_fn_name="scaled_mse",
        eos_token_id = tokenizer.eos_token_id
    )

    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path, config)
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if (not use_ddp) or (local_rank == 0):
        decoded_preds, decoded_targets = text_generation(model, config, device)
        if decoded_preds and decoded_targets and ((not use_ddp) or (local_rank == 0)):
            compute_text_metrics(decoded_preds, decoded_targets)
    
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
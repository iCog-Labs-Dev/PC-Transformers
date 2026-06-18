import torch
from tokenizers import Tokenizer
from predictive_coding.config import GPTConfig
from utils.model_utils import load_model, decode_ids, compute_text_metrics
from utils.config_utils import load_best_config
from utils.model_utils import set_seed
import torch.nn.functional as F
import torch.distributed as dist
from utils.device_utils import setup_device
import argparse
from data_preparation.config import vocab_size

"""
This script generates text using the trained predictive coding transformer model.
It generates new tokens from scratch without a prompt.

Usage: torchrun --nproc-per-node=<NUM_GPU> generate_text.py

"""
local_rank, device, use_distributed = setup_device()

def generate_text(model, config, max_new_tokens, temperature, device=None, use_cache=True, tokenizer=None):
    model.eval()
    
    # Start with just a BOS token or empty sequence
    # Using a start token (0) as the initial input
    input_tensor = torch.tensor([[0]], device=device)  # Start with token 0 (often BOS)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_tensor, input_tensor, generate=True)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tensor = torch.cat((input_tensor, next_token), dim=1)

    return input_tensor[0]

def text_generation(model, config, device=None, max_samples=2, max_new_tokens=100, use_cache=True):
    decoded_preds = []

    tokenizer = Tokenizer.from_file("data_preparation/tokenizer.json")
    
    for sample_idx in range(max_samples):
        if hasattr(model, 'reset_rope_cache'):
            model.reset_rope_cache()
        if hasattr(model, 'modules'):
            for module in model.modules():
                if hasattr(module, 'clear_kv_cache'):
                    module.clear_kv_cache()
        
        generated_ids = generate_text(
            model, config,
            max_new_tokens=max_new_tokens,
            temperature=0.8, device=device,
            use_cache=use_cache,
            tokenizer=tokenizer
        )
        generated_str = decode_ids(tokenizer, generated_ids.tolist(), stop_at_eos=True)

        print(f"\n[Sample {sample_idx + 1}]")
        print(f"[GENERATED]: {generated_str}")

        decoded_preds.append(generated_str)

    return decoded_preds, None

def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention for attention layers')
    args = parser.parse_args()

    if use_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    print(f"[Rank {local_rank}] Using device: {device}")
    
    best_config = load_best_config()
    max_new_tokens = 200

    config = GPTConfig(
        vocab_size = vocab_size,
        block_size = best_config["block_size"] + max_new_tokens,
        lr = best_config["peak_learning_rate"],
        inference_lr = best_config["inference_lr"],
        peak_learning_rate = best_config["peak_learning_rate"],
        warmup_steps = best_config["warmup_steps"],
        n_embed = best_config["n_embed"],
        dropout = best_config["dropout"],
        T = best_config["T"],
        num_heads = best_config["num_heads"],
        n_blocks = best_config["n_blocks"],
        batch_size = best_config["batch_size"],
        num_epochs = best_config["num_epochs"], 
        internal_energy_fn_name=best_config["internal_energy_fn_name"],
        output_energy_fn_name=best_config["output_energy_fn_name"],
        combined_internal_weight=best_config["combined_internal_weight"],
        combined_output_weight=best_config["combined_output_weight"],
        use_flash_attention=best_config["use_flash_attention"],
        alpha = best_config["alpha"],
        clamp_value = best_config["clamp_value"],
        clip_value = best_config["clip_value"],
        optimizer_name = best_config["optimizer_name"],
        output_optimizer_name = best_config.get("output_optimizer_name", "adam"),
        optimizer_beta1 = best_config["optimizer_beta1"],
        optimizer_beta2 = best_config["optimizer_beta2"],
        optimizer_eps = best_config["optimizer_eps"],
        optimizer_momentum = best_config.get("optimizer_momentum", 0.9),
        optimizer_weight_decay = best_config.get("optimizer_weight_decay", 0.1),
    )
    
    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path, config)
    model = model.to(device)
    
    if use_distributed:
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    if not dist.is_initialized() or dist.get_rank() == 0:
        decoded_preds, decoded_targets = text_generation(model, config, device, max_samples=2, use_cache=True)
    
    if use_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
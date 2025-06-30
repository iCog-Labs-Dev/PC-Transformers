import torch
from predictive_coding.config import GPTConfig
from utils.model_utils import load_tokenizer, load_model, reset_pc_modules, decode_ids, compute_text_metrics
import torch.nn.functional as F
from Data_preprocessing.dataloader import get_loaders

"""
Usage: python generate_text.py

This script generates text using a trained predictive coding transformer model.
It takes a prompt, generates new tokens, and prints the prompt, target, and generated text.
"""

def generate_text(model, config, input_ids, max_new_tokens, temperature, device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
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

def text_generation(model, config, device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    decoded_preds = []
    decoded_targets = []
    num_samples = 5
    prompt_len = 5
    
    _, _, test_loader = get_loaders()
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
        
        print(f"\n[Batch {batch_idx + 1}, Sample {i + 1}]")
        print(f"[PROMPT ]: {prompt_str}")
        print(f"[TARGET ]: {target_str}")
        print(f"[PREDICT]: {generated_str}")

    return decoded_preds, decoded_targets

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    config = GPTConfig(
        vocab_size = vocab_size,
        block_size=256,
        n_embed=64,
        dropout=0.1,
        local_learning_rate=1e-5,
        T=2,
        is_holding_error=True,
        num_heads=2,
        n_blocks=4,
        num_epochs=1,
        update_bias=True,
        energy_fn_name="kld",
        eos_token_id = tokenizer.token_to_id("[EOS]")
    )

    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path, config)
    decoded_preds, decoded_targets = text_generation(model, config, device)
    if decoded_preds and decoded_targets:
        compute_text_metrics(decoded_preds, decoded_targets)
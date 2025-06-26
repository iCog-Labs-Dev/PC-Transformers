import torch
from predictive_coding.config import GPTConfig
from utils.model_utils import load_tokenizer, load_model, reset_pc_modules, decode_ids
import torch.nn.functional as F
from Data_preprocessing.dataloader import test_loader

"""
Usage: python generate_text.py

This script generates text using a trained predictive coding transformer model.
It takes a prompt, generates new tokens, and prints the prompt, target, and generated text.
"""

def generate_text(model, config, input_ids, max_new_tokens=50, temperature=1.0):
    model.eval()
    input_tensor = input_ids.unsqueeze(0)

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

tokenizer = load_tokenizer()
vocab_size = tokenizer.get_vocab_size()
pad_token_id = tokenizer.token_to_id("[PAD]")

config = GPTConfig(
    vocab_size = vocab_size,
    block_size=320,
    peak_learning_rate= 1.51e-04,
    warmup_steps= 94,
    n_embed=464,
    dropout=0.2572947974079954,
    local_learning_rate= 0.0,
    T=8,
    is_holding_error=True,
    num_heads=16,
    n_blocks=6,
    num_epochs=1,
    update_bias=False,
    energy_fn_name="mse",
    eos_token_id = tokenizer.token_to_id("[EOS]")
)

model_path = "checkpoints/pc_transformer.pt"
model = load_model(model_path, config)

for batch_idx, batch in enumerate(test_loader):
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    break 

num_samples = 5
prompt_len = 5
i = 64

for i in range(num_samples):
    prompt_ids = input_ids[i][:prompt_len]
    generated_ids = generate_text(model, config, prompt_ids, max_new_tokens= 50, temperature=0.7)

    target_continuation = target_ids[i][prompt_len:]
    target_continuation = target_continuation[target_continuation != pad_token_id].tolist()

    generated_continuation = generated_ids[prompt_len:].tolist()

    # Decode all
    prompt_str = decode_ids(tokenizer, prompt_ids.tolist())
    target_str = decode_ids(tokenizer, target_continuation, stop_at_eos=True)
    predict_str = decode_ids(tokenizer, generated_continuation, stop_at_eos=True)

    print(f"\n[Batch {batch_idx + 1}, Sample {i + 1}]")
    print(f"[PROMPT ]: {prompt_str}")
    print(f"[TARGET ]: {target_str}")
    print(f"[PREDICT]: {predict_str}")

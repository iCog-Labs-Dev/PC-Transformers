import torch
from predictive_coding.config import GPTConfig
from utils.model_utils import load_tokenizer, load_model, reset_pc_modules
import torch.nn.functional as F
from Data_preprocessing.dataloader import test_loader

"""
Usage: python generate_text.py

This script generates text using a trained predictive coding transformer model.
It takes a prompt, generates new tokens, and prints the prompt, target, and generated text.
"""

def generate_text(input_ids, max_new_tokens=50, temperature=1.0):
    """
    Generate text from a prompt using the trained model.

    Args:
        input_ids (torch.Tensor): Tensor of shape (prompt_length,) with initial token IDs.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature for output distribution.

    Returns:
        torch.Tensor: Tensor of shape (prompt_length + max_new_tokens,) with generated token IDs.
    """
    model.eval()
    input_tensor = input_ids.unsqueeze(0)

    for _ in range(max_new_tokens):
        if input_tensor.size(1) > config.block_size:
            input_tensor = input_tensor[:, -config.block_size:]
        with torch.no_grad():
            logits = model(input_tensor, input_tensor)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tensor = torch.cat((input_tensor, next_token), dim=1)
        
        reset_pc_modules(model)
                
    return input_tensor[0] 

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
    energy_fn_name="kld"
)

model_path = "checkpoints/pc_transformer.pt"
model = load_model(model_path, config)

for batch in test_loader:
    input_ids = batch["input_ids"][0]
    target_ids = batch["target_ids"][0]
    break

prompt_length = 10
prompt_ids = input_ids[:prompt_length]
generated_ids = generate_text(prompt_ids, max_new_tokens=50, temperature=0.7)

print("\n Prompt:")
print(tokenizer.decode(prompt_ids.tolist()))

print("\n Target")
print(tokenizer.decode(target_ids.tolist()))

print("\n Generated Text:")
print(tokenizer.decode(generated_ids.tolist()))
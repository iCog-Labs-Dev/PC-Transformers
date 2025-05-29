import os
import time
import torch
from tokenizers import Tokenizer
from model_architecture.pc_t_model import PCTransformer
from predictive_coding.config import GPTConfig
from Data_preprocessing.dataloader import test_loader
from Data_preprocessing.config import Config
import torch.nn.functional as F

def load_model(model_path, config):
    model = PCTransformer(config)
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate(model, dataloader, device):
    start_time = time.time()
    model.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)

            logits = model.evaluate(input_ids)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )

            total_loss += loss.item()
            batch_count += 1

    average_loss = total_loss / batch_count
    perplexity = torch.exp(torch.tensor(average_loss)) 

    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f}s")
    print(f"Cross-Entropy Loss: {average_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
  

    return average_loss

def generate_text(input_ids, max_new_tokens=50, temperature=1.0):
    input_tensor = input_ids.unsqueeze(0)

    for _ in range(max_new_tokens):
        if input_tensor.size(1) > config.block_size:
            input_tensor = input_tensor[:, -config.block_size:]

        logits = model.evaluate(input_tensor)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tensor = torch.cat((input_tensor, next_token), dim=1)

    return input_tensor[0]

tokenizer_path = os.path.join(Config.TOKENIZER_DIR, "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

config = GPTConfig(
    vocab_size = vocab_size,
    block_size=256,
    n_embed=64,
    dropout=0.1,
    local_learning_rate=1e-7,
    T=1,
    is_holding_error=True,
    num_heads=2,
    n_blocks=2,
    num_epochs=1,
    update_bias=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "checkpoints/pc_transformer.pt"
model = load_model(model_path, config)
model.to(device)
evaluate(model, test_loader, device)

# Generate text using the trained model
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
print(tokenizer.decode(input_ids.tolist()))

print("\n Generated Text:")
print(tokenizer.decode(generated_ids.tolist()))

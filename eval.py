import os
import time
import torch
from tokenizers import Tokenizer
from model_architecture.pc_t_model import PCTransformer
from model_architecture.transformer_utils import ids_to_one_hot, energy_fn
from predictive_coding.config import GPTConfig
from Data_preprocessing.dataloader import train_loader
from Data_preprocessing.config import Config
from predictive_coding.pc_utils import energy_fn

def load_model(model_path, config):
    model = PCTransformer(config)
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate(model, dataloader, config):
    start_time = time.time()
    
    model.eval()
    total_energy = 0.0
    batch_count = 0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            targets = batch["targets"]

            output = model(input_ids)
            targets = ids_to_one_hot(targets, config.vocab_size)

            energy = energy_fn(output, targets).mean()
            total_energy += energy.item()
            batch_count += 1

            pred_ids = torch.argmax(output, dim=-1)
            true_ids = torch.argmax(targets, dim=-1)
            correct_preds += (pred_ids == true_ids).sum().item()
            total_preds += pred_ids.numel()

    avg_energy = total_energy / batch_count
    accuracy = (correct_preds / total_preds) * 100
    elapsed_time = time.time() - start_time


    print(f"\n========= Evaluation Results =========")
    print(f"Average Predictive Coding Energy: {avg_energy:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    return avg_energy


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

model_path = "checkpoints/pc_transformer.pt"
model = load_model(model_path, config)
evaluate(model, train_loader, config)

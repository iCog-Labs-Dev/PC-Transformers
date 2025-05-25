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
    total_preds = 0
    correct_preds = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)

            logits = model.evaluate(input_ids)

            # Flatten logits and targets for loss computation
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
                reduction='sum'  # Sum total loss for all tokens
            )

            total_loss += loss.item()

            # Compute accuracy
            preds = torch.argmax(logits, dim=-1)
            mask = targets != 0  # Ignore padding tokens
            correct = (preds == targets) & mask
            correct_preds += correct.sum().item()
            total_preds += mask.sum().item()

    average_loss = total_loss / total_preds
    average_accuracy = correct_preds / total_preds

    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f}s")
    print(f"Correct Predictions: {correct_preds}")
    print(f"Total Predictions: {total_preds}")
    print(f"Cross-Entropy Loss: {average_loss:.4f}")
    print(f"Accuracy: {average_accuracy * 100:.2f}%")

    return average_loss

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

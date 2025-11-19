import torch
import time
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import train_path, valid_path, test_path, tokenizer_path, encoded_dir, vocab_size, special_tokens

"""
Usage: python prepare_tokens.py
"""
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def build_tokenizer():
    """ This function trains a BPE tokenizer on the given dataset and saves it."""
    tokenizer = Tokenizer(BPE(unk_token = "<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        special_tokens = special_tokens,
        vocab_size = vocab_size
        )

    paths = [train_path, valid_path, test_path]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

    start_time = time.perf_counter()
    tokenizer.train(files=[str(train_path)], trainer=trainer)
    elapsed = time.perf_counter() - start_time

    tokenizer.save(str(tokenizer_path))
    logger.info(f"Tokenizer saved at {tokenizer_path}")
    logger.info(f"Tokenizer training took {elapsed:.2f} seconds.")

    return tokenizer

def encode_and_save(tokenizer):
    """
    This function takes the raw text from train/valid/test, uses the tokenizer to convert words 
    (or subwords) into token IDs, and saves them as PyTorch tensors (.pt files).
    """
    encoded_dir.mkdir(exist_ok=True, parents=True)

    splits = {
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
    }

    for split_name, path in splits.items():
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        start_time = time.perf_counter()
        encoded_ids = tokenizer.encode(text).ids  
        elapsed = time.perf_counter() - start_time

        tensor = torch.tensor(encoded_ids, dtype=torch.long)
        save_path = encoded_dir / f"{split_name}.pt"
        torch.save(tensor, save_path)

        logger.info(f"Saved encoded {split_name} dataset in {save_path}")
        logger.info(f"Tokenizing {split_name} split took {elapsed:.2f} seconds.")

if __name__ == "__main__":
    tokenizer = build_tokenizer()
    encode_and_save(tokenizer)
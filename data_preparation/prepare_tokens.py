# data_preparation/prepare_tokens.py
import torch
import time
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import tiktoken

"""
Usage: python prepare_tokens.py
This will use the TOKENIZER_NAME from config.py
"""

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import from config
from config import (
    train_path, valid_path, test_path, 
    vocab_size, special_tokens,
    TOKENIZER_NAME, ENCODED_DIR, TOKENIZER_PATH
)

def build_bpe_tokenizer():
    """Train a BPE tokenizer on the given dataset and save it."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        special_tokens=special_tokens,
        vocab_size=vocab_size 
    )

    paths = [train_path, valid_path, test_path]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

    start_time = time.perf_counter()
    tokenizer.train(files=[str(p) for p in paths], trainer=trainer)
    elapsed = time.perf_counter() - start_time

    tokenizer.save(str(TOKENIZER_PATH))
    logger.info(f"BPE Tokenizer saved at {TOKENIZER_PATH}")
    logger.info(f"BPE Tokenizer training took {elapsed:.2f} seconds.")
    logger.info(f"BPE Vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer

def get_tiktoken_tokenizer():
    """Load the tiktoken tokenizer."""
    start_time = time.perf_counter()
    tokenizer = tiktoken.get_encoding("cl100k_base")
    elapsed = time.perf_counter() - start_time
    
    logger.info(f"Tiktoken tokenizer loaded in {elapsed:.2f} seconds.")
    logger.info(f"Tiktoken vocab size: {tokenizer.n_vocab}")
    
    # Save tokenizer info (just for reference)
    import json
    tokenizer_info = {
        "name": "cl100k_base",
        "vocab_size": tokenizer.n_vocab,
        "type": "tiktoken"
    }
    with open(TOKENIZER_PATH, "w") as f:
        json.dump(tokenizer_info, f, indent=2)
    
    return tokenizer

def encode_and_save(tokenizer):
    """Encode datasets using the tokenizer and save as PyTorch tensors."""
    ENCODED_DIR.mkdir(exist_ok=True, parents=True)

    splits = {
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
    }

    for split_name, path in splits.items():
        if not path.exists():
            logger.warning(f"File not found: {path}, skipping {split_name}")
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        start_time = time.perf_counter()
        
        # Handle different tokenizer APIs
        if TOKENIZER_NAME == "bpe":
            encoded_ids = tokenizer.encode(text).ids
        else:  # tiktoken
            encoded_ids = tokenizer.encode(text)
            
        elapsed = time.perf_counter() - start_time

        tensor = torch.tensor(encoded_ids, dtype=torch.long)
        save_path = ENCODED_DIR / f"{split_name}.pt"
        torch.save(tensor, save_path)

        logger.info(f"Saved encoded {split_name} dataset: {save_path}")
        logger.info(f"  - Tokens: {len(encoded_ids):,}")
        logger.info(f"  - Range: [{min(encoded_ids)}, {max(encoded_ids)}]")
        logger.info(f"  - Time: {elapsed:.2f}s")

if __name__ == "__main__":
    logger.info(f"\n{'='*60}")
    logger.info(f"Preparing {TOKENIZER_NAME.upper()} tokenizer")
    logger.info(f"Output directory: {ENCODED_DIR}")
    logger.info(f"{'='*60}")
    
    if TOKENIZER_NAME == "bpe":
        tokenizer = build_bpe_tokenizer()
    else:  # tiktoken
        tokenizer = get_tiktoken_tokenizer()
    
    logger.info(f"\nEncoding datasets...")
    encode_and_save(tokenizer)
    
    logger.info(f"\n{TOKENIZER_NAME.upper()} tokenization complete!")
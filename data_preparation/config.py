# data_preparation/config.py
from pathlib import Path
import tiktoken

TOKENIZER_NAME = "tiktoken"  # Options: "bpe" or "tiktoken"

base_dir = Path(__file__).resolve().parent.parent

# Dataset paths
data_dir = base_dir / "data_preparation" / "data" / "tiny_shakespear"
train_path = data_dir / "train.csv"
valid_path = data_dir / "validation.csv"
test_path = data_dir / "test.csv"

# Tokenizer parameters for BPE
special_tokens = ["[UNK]", "[BOS]", "[EOS]", "[PAD]"]

# Set vocab size and pad token ID based on tokenizer name
if TOKENIZER_NAME == "bpe":
    vocab_size = 11711
    PAD_TOKEN_ID = 3
    ENCODED_DIR = base_dir / "data_preparation" / "encoded_bpe"
    TOKENIZER_PATH = base_dir / "data_preparation" / "tokenizer_bpe.json"
else:  # tiktoken
    # Get the actual tiktoken vocab size dynamically
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        vocab_size = tokenizer.n_vocab  # This is 100277 but includes special tokens
        # The actual max token ID might be higher
        # Test with a sample to find the actual max ID
        test_tokens = tokenizer.encode("test")
        actual_max_id = max(test_tokens)
        # Get the real max ID by encoding special tokens
        special_tokens_sample = tokenizer.encode("<|endoftext|><|fim_prefix|><|fim_middle|><|fim_suffix|>")
        actual_max_id = max(actual_max_id, max(special_tokens_sample))
        vocab_size = actual_max_id + 1  # Set vocab size to accommodate the highest token ID
    except:
        vocab_size = 100277  # Fallback
    
    PAD_TOKEN_ID = 100257  # <|endoftext|>
    ENCODED_DIR = base_dir / "data_preparation" / "encoded_tiktoken"
    TOKENIZER_PATH = base_dir / "data_preparation" / "tokenizer_tiktoken.json"

# For backward compatibility with existing code
encoded_dir = ENCODED_DIR
tokenizer_path = TOKENIZER_PATH
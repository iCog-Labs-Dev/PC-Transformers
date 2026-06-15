import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader, DistributedSampler
from data_preparation.dataset import EncodedDataset
from data_preparation.config import ENCODED_DIR, vocab_size, PAD_TOKEN_ID, TOKENIZER_NAME

def get_datasets(block_size: int, stride: int = 1):
    """Load train, validation, and test datasets from encoded token ID files."""
    # Check if encoded files exist
    train_file = ENCODED_DIR / "train.pt"
    if not train_file.exists():
        raise FileNotFoundError(
            f"Tokenized file not found: {train_file}\n"
            f"Please run: python data_preparation/prepare_tokens.py"
        )
    
    print(f"Loading datasets from: {ENCODED_DIR}")
    print(f"Using tokenizer: {TOKENIZER_NAME}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Pad token ID: {PAD_TOKEN_ID}")
    
    # Pass vocab_size to dataset for validation
    train_dataset = EncodedDataset(
        train_file, 
        block_size, 
        stride=stride,
        pad_token_id=PAD_TOKEN_ID,
        vocab_size=vocab_size
    )
    valid_dataset = EncodedDataset(
        ENCODED_DIR / "valid.pt", 
        block_size, 
        stride=stride,
        pad_token_id=PAD_TOKEN_ID,
        vocab_size=vocab_size
    )
    test_dataset = EncodedDataset(
        ENCODED_DIR / "test.pt", 
        block_size, 
        stride=stride,
        pad_token_id=PAD_TOKEN_ID,
        vocab_size=vocab_size
    )
    
    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Valid sequences: {len(valid_dataset):,}")
    print(f"Test sequences: {len(test_dataset):,}")
    
    return train_dataset, valid_dataset, test_dataset

def get_loaders(batch_size: int, block_size: int, distributed: bool = False, stride: int = 1):
    """Wrap datasets into PyTorch DataLoaders with batching and shuffling."""
    train_dataset, valid_dataset, test_dataset = get_datasets(
        block_size=block_size,
        stride=block_size,
    )
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = valid_sampler = test_sampler = None

    train_loader = DataLoader(
        train_dataset, 
        batch_size= batch_size, 
        sampler=train_sampler,
        shuffle=(train_sampler is None), 
        drop_last=False
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size= batch_size,
        sampler=valid_sampler,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size= batch_size,
        sampler=test_sampler,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader
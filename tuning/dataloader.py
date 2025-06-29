import torch
import psutil
from torch.utils.data import Subset, DataLoader
from Data_preprocessing.dataloader import get_loaders
from utils.model_utils import pad_collate_fn, load_tokenizer

def get_optimal_data_sizes():
    """Determine optimal data sizes based on available memory"""
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return (20000, 5000) if mem_gb >= 8 else (2000, 400) if mem_gb >= 4 else (1200, 240)
    else:
        mem_gb = psutil.virtual_memory().total / (1024**3)
        return (1500, 300) if mem_gb >= 16 else (800, 160)

def create_subset_loaders(batch_size):
    """Create appropriately sized data loaders"""
    tokenizer = load_tokenizer()
    pad_token_id = tokenizer.pad_token_id
    train_loader, valid_loader, _ = get_loaders()

    train_size, valid_size = get_optimal_data_sizes()
    max_train = len(train_loader.dataset)
    max_valid = len(valid_loader.dataset)

    train_indices = torch.randperm(max_train)[:min(train_size, max_train)]
    valid_indices = torch.randperm(max_valid)[:min(valid_size, max_valid)]

    train_subset = Subset(train_loader.dataset, train_indices)
    valid_subset = Subset(valid_loader.dataset, valid_indices)

    train_subset_loader = DataLoader(train_subset, batch_size=batch_size,
        shuffle=True, collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id))

    valid_subset_loader = DataLoader(valid_subset, batch_size=batch_size,
        shuffle=False, collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id))

    return train_subset_loader, valid_subset_loader

def get_dynamic_batch_size(n_embed, block_size):
    """Calculate optimal batch size based on model size"""
    if torch.cuda.is_available():
        memory = torch.cuda.get_device_properties(0).total_memory
        usable_mem = memory - 1.5 * (1024**3) 
        sequence_mem = block_size * n_embed * 4
        return max(4, min(24, int(usable_mem / (sequence_mem * 3000))))
    else:
        return max(4, min(12, 8))
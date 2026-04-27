import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader, DistributedSampler
from data_preparation.config import encoded_dir
from data_preparation.dataset import EncodedDataset

def get_datasets(block_size: int, stride: int = 1):
    """ Load train, validation, and test datasets from encoded token ID files."""
    train_dataset = EncodedDataset(encoded_dir/"train.pt", block_size, stride=stride)
    valid_dataset = EncodedDataset(encoded_dir/"valid.pt", block_size, stride=stride)
    test_dataset = EncodedDataset(encoded_dir/"test.pt", block_size, stride=stride)
    
    return train_dataset, valid_dataset, test_dataset

def get_loaders(batch_size: int, block_size: int, distributed: bool = False, stride: int = 1):
    """Wrap datasets into PyTorch DataLoaders with batching and shuffling."""
    train_dataset, valid_dataset, test_dataset = get_datasets(
        block_size=block_size,
        stride=stride,
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
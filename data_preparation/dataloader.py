import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader, DistributedSampler, Subset
from data_preparation.config import encoded_dir, max_len, batch_size
from data_preparation.dataset import EncodedDataset


MAX_BATCHES = 20

def get_datasets():
    train_dataset = EncodedDataset(encoded_dir/"train.pt", max_len)
    valid_dataset = EncodedDataset(encoded_dir/"valid.pt", max_len)
    test_dataset = EncodedDataset(encoded_dir/"test.pt", max_len)
    
    return train_dataset, valid_dataset, test_dataset


def limit_dataset(dataset):
    max_samples = batch_size * MAX_BATCHES
    indices = list(range(min(len(dataset), max_samples)))
    return Subset(dataset, indices)


def get_loaders(distributed: bool = False):
    train_dataset, valid_dataset, test_dataset = get_datasets()

    # 🔥 Limit datasets to 50 batches
    train_dataset = limit_dataset(train_dataset)
    valid_dataset = limit_dataset(valid_dataset)
    test_dataset = limit_dataset(test_dataset)

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = valid_sampler = test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader
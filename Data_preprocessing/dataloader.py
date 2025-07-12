from torch.utils.data import DataLoader, DistributedSampler
from Data_preprocessing.datasets.penn_treebank import PennTreebankDataset
from Data_preprocessing.config import Config
from utils.model_utils import pad_collate_fn, load_tokenizer
from functools import partial

def get_datasets():
    train_dataset = PennTreebankDataset("train_ids.pkl", Config.tokenizer_dir, Config.max_length)
    valid_dataset = PennTreebankDataset("valid_ids.pkl", Config.tokenizer_dir, Config.max_length)
    test_dataset = PennTreebankDataset("test_ids.pkl", Config.tokenizer_dir, Config.max_length)

    return train_dataset, valid_dataset, test_dataset

def collate_fn_with_pad(batch, pad_token_id):
    return pad_collate_fn(batch, pad_token_id)

def get_loaders(distributed: bool = False):
    tokenizer = load_tokenizer()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    train_dataset, valid_dataset, test_dataset = get_datasets()

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = valid_sampler = test_sampler = None

    collate = partial(collate_fn_with_pad, pad_token_id=pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=Config.num_workers > 0,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=Config.num_workers > 0
    )                     
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=Config.num_workers > 0
    )
    return train_loader, valid_loader, test_loader

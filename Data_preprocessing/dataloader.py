
from torch.utils.data import DataLoader, DistributedSampler
from Data_preprocessing.datasets.merged_sets import TokenizedDataset
from Data_preprocessing.config import Config
from utils.model_utils import pad_collate_fn, load_tokenizer

def get_datasets():
    train_dataset = TokenizedDataset("train", Config.TOKENIZER_DIR, Config.MAX_LENGTH)
    valid_dataset = TokenizedDataset("valid", Config.TOKENIZER_DIR, Config.MAX_LENGTH)
    test_dataset = TokenizedDataset("test", Config.TOKENIZER_DIR, Config.MAX_LENGTH)

    return train_dataset, valid_dataset, test_dataset

def get_loaders(distributed: bool = False):
    tokenizer = load_tokenizer()
    pad_token_id = tokenizer.pad_token_id
    train_dataset, valid_dataset, test_dataset = get_datasets()
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = valid_sampler = test_sampler = None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        sampler=train_sampler,
        shuffle=(train_sampler is None),  
        num_workers=Config.num_workers,
        pin_memory=True,                            
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id),
        persistent_workers=Config.num_workers > 0,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=Config.BATCH_SIZE,
        sampler=valid_sampler, 
        shuffle = False,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id),
        persistent_workers=Config.num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        sampler=test_sampler,
        shuffle = False,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id),
        persistent_workers=Config.num_workers > 0)

    return train_loader, valid_loader, test_loader
from torch.utils.data import DataLoader, Subset
from .datasets.penn_treebank import PennTreebankDataset
from .config import Config
from utils.model_utils import pad_collate_fn, load_tokenizer

train_dataset = PennTreebankDataset("train_ids.pkl", Config.TOKENIZER_DIR, Config.MAX_LENGTH)
train_dataset = Subset(train_dataset, range(min(len(train_dataset), 100000)))
valid_dataset = PennTreebankDataset("valid_ids.pkl", Config.TOKENIZER_DIR, Config.MAX_LENGTH)
test_dataset = PennTreebankDataset("test_ids.pkl", Config.TOKENIZER_DIR, Config.MAX_LENGTH)
test_dataset = Subset(test_dataset, range(min(len(test_dataset), 25000)))

tokenizer = load_tokenizer()
pad_token_id = tokenizer.token_to_id("[PAD]")
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=64,
    collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id)
)


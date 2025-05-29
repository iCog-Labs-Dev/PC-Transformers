from torch.utils.data import DataLoader, Subset
from .datasets.penn_treebank import PennTreebankDataset
from .config import Config

train_dataset = PennTreebankDataset("train_ids.pkl", Config.TOKENIZER_DIR, Config.MAX_LENGTH)
train_dataset = Subset(train_dataset, range(0, 50000))
valid_dataset = PennTreebankDataset("valid_ids.pkl", Config.TOKENIZER_DIR, Config.MAX_LENGTH)
test_dataset = PennTreebankDataset("test_ids.pkl", Config.TOKENIZER_DIR, Config.MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size= 64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size= 64)
test_loader = DataLoader(test_dataset, batch_size= 64)


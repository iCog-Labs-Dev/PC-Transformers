import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class PennTreebankDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        encoding = self.tokenizer.encode(text)
        self.tokens = encoding.ids

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.tokens[idx : idx + self.block_size], dtype=torch.long)
        target_ids = torch.tensor(self.tokens[idx + 1 : idx + 1 + self.block_size], dtype=torch.long)
        return {"input_ids": input_ids, "target_ids": target_ids}

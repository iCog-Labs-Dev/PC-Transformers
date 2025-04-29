import torch
import pickle
import os
from torch.utils.data import Dataset

class PennTreebankDataset(Dataset):
    def __init__(self, tokenized_file, tokenizer_dir, block_size):
        """
        Args:
            tokenized_file (str): Path to the tokenized dataset file (e.g., train_ids.pkl).
            tokenizer_dir (str): Directory where the tokenizer and tokenized files are stored.
            block_size (int): The size of each input sequence (number of tokens).
        """
        self.tokenizer_dir = tokenizer_dir
        self.block_size = block_size

        tokenized_file_path = os.path.join(self.tokenizer_dir, tokenized_file)
        if not os.path.exists(tokenized_file_path):
            raise FileNotFoundError(f"Tokenized file not found: {tokenized_file_path}")

        with open(tokenized_file_path, 'rb') as f:
            self.tokens = pickle.load(f)

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size

        input_ids = torch.tensor(self.tokens[start:end], dtype=torch.long)
        targets = torch.tensor(self.tokens[start+1:end+1], dtype=torch.long)

        return {"input_ids": input_ids, "targets": targets}
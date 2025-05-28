import torch
import pickle
import os
from torch.utils.data import Dataset

class PennTreebankDataset(Dataset):
    def __init__(self, tokenized_file, tokenizer_dir, block_size):
        """
        Args:
            tokenized_file: path to the tokenized dataset file (e.g., train_ids.pkl).
            tokenizer_dir: directory where the tokenizer and tokenized files are stored.
            block_size: size of each input sequence (number of tokens).
        """
        self.tokenizer_dir = tokenizer_dir
        self.block_size = block_size

        tokenized_file_path = os.path.join(self.tokenizer_dir, tokenized_file)
        if not os.path.exists(tokenized_file_path):
            raise FileNotFoundError(
                f"Tokenized file not found: {tokenized_file_path}\n"
                "Please tokenize the dataset first by running:\n\n"
                " python3 -m Data_preprocessing.tokenizer.bpe_tokenizer\n"
            )

        with open(tokenized_file_path, 'rb') as f:
            self.tokens = pickle.load(f)

        if isinstance(self.tokens[0], list):
            self.tokens = [token for seq in self.tokens for token in seq]

        self.num_sequences = len(self.tokens) - self.block_size

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.tokens[idx : idx + self.block_size], dtype=torch.long)
        target_ids = torch.tensor(self.tokens[idx + 1 : idx + 1 + self.block_size], dtype=torch.long)

        return {"input_ids": input_ids, "target_ids": target_ids}
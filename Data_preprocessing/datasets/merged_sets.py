import torch
import pickle
import os
from torch.utils.data import Dataset
from Data_preprocessing.config import Config

class TokenizedDataset(Dataset):
    def __init__(self, subset_name, tokenizer_dir, block_size):
        """
        Args:
            subset_name: Name of the dataset subset (e.g., "train", "valid", "test").
            tokenizer_dir: Directory where the tokenizer and tokenized files are stored.
            block_size: Size of each input sequence (number of tokens).
        """
        self.tokenizer_dir = tokenizer_dir
        self.block_size = block_size
        tokenized_file_path = os.path.join(self.tokenizer_dir, f"{Config.DATASET_NAME}_{subset_name}_ids.pkl")
        if not os.path.exists(tokenized_file_path):
            raise FileNotFoundError(
                f"Tokenized file not found: {tokenized_file_path}\n"
                "Please tokenize the dataset first by running:\n"
                " python -m Data_preprocessing.tokenizer.gpt2_tokenizer\n"
            )
        with open(tokenized_file_path, 'rb') as f:
            self.sequences = pickle.load(f)
        self.sequences = [seq for seq in self.sequences if len(seq) > 1]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1][:self.block_size], dtype=torch.long)
        target_ids = torch.tensor(seq[1:][:self.block_size], dtype=torch.long)
        return {"input_ids": input_ids, "target_ids": target_ids}
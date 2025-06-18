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
            self.sequences = pickle.load(f)

        self.sequences = [seq for seq in self.sequences if len(seq) > 1]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1][:self.block_size], dtype=torch.long)
        target_ids = torch.tensor(seq[1:][:self.block_size], dtype=torch.long)
        
        return {"input_ids": input_ids, "target_ids": target_ids}
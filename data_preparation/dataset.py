import torch
from pathlib import Path
from torch.utils.data import Dataset

class EncodedDataset(Dataset):
    """ Dataset that splits token ID tensors into input-target sequences for next-token prediction."""
    def __init__(self, file_path, block_size):
        self.block_size = block_size

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Tokenized file not found: {file_path}")
        
        tokens = torch.load(file_path, weights_only=False)

        total_len = (len(tokens) // (block_size + 1)) * (block_size + 1)
        tokens = tokens[:total_len]
        self.sequences = tokens.view(-1, block_size + 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = seq[:-1].clone().detach()
        target_ids = seq[1:].clone().detach()

        return {"input_ids": input_ids, "target_ids": target_ids}
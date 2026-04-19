import torch
from pathlib import Path
from torch.utils.data import Dataset

class EncodedDataset(Dataset):
    """ Dataset that splits token ID tensors into input-target sequences for next-token prediction using sliding window."""
    def __init__(self, file_path, block_size, stride=None):
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Tokenized file not found: {file_path}")
        
        tokens = torch.load(file_path, weights_only=False)

        sequences = []
        for i in range(0, len(tokens) - block_size, self.stride):
            sequences.append(tokens[i:i + block_size + 1])
        self.sequences = torch.stack(sequences) if sequences else torch.tensor([])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = seq[:-1].clone().detach()
        target_ids = seq[1:].clone().detach()

        return {"input_ids": input_ids, "target_ids": target_ids}
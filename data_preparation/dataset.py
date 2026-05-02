import torch
from pathlib import Path
from torch.utils.data import Dataset

class EncodedDataset(Dataset):
    """ Dataset that splits token ID tensors into input-target sequences for next-token prediction using sliding window."""
    def __init__(self, file_path, block_size, stride=None):
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")

        self.block_size = block_size
        self.stride = 1 if stride is None else stride
        if self.stride < 1:
            raise ValueError("stride must be at least 1.")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Tokenized file not found: {file_path}")
        
        tokens = torch.load(file_path, weights_only=False)
        window_size = block_size + 1

        if len(tokens) < window_size:
            self.sequences = torch.empty(
                (0, window_size),
                dtype=tokens.dtype,
                device=tokens.device,
            )
        else:
            # `unfold` keeps overlapping windows as a view instead of copying the corpus.
            self.sequences = tokens.unfold(0, window_size, self.stride)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = seq[:-1].clone().detach()
        target_ids = seq[1:].clone().detach()

        return {"input_ids": input_ids, "target_ids": target_ids}

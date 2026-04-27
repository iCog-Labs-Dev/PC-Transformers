import torch
from pathlib import Path
from torch.utils.data import Dataset

class EncodedDataset(Dataset):
<<<<<<< HEAD
    """ Dataset that splits token ID tensors into input-target sequences for next-token prediction using sliding window."""
    def __init__(self, file_path, block_size, stride=None):
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")

=======
    """ Dataset that splits token ID tensors into input-target sequences for next-token prediction, with padding."""
    def __init__(self, file_path, block_size, pad_token_id=3):
>>>>>>> a53d33d (Optimize data loading: Implement sequence padding and disable drop_last to prevent data loss on small datasets.)
        self.block_size = block_size
        self.stride = 1 if stride is None else stride
        if self.stride < 1:
            raise ValueError("stride must be at least 1.")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Tokenized file not found: {file_path}")
        
        tokens = torch.load(file_path, weights_only=False)
        window_size = block_size + 1


        # Calculate how much padding we need to avoid throwing away data
        chunk_size = block_size + 1
        remainder = len(tokens) % chunk_size
        
        if remainder > 0:
            padding_len = chunk_size - remainder
            padding = torch.full((padding_len,), pad_token_id, dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding])
            
        self.sequences = tokens.view(-1, chunk_size)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = seq[:-1].clone().detach()
        target_ids = seq[1:].clone().detach()

        return {"input_ids": input_ids, "target_ids": target_ids}

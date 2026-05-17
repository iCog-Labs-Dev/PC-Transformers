import torch
from pathlib import Path
from torch.utils.data import Dataset

class EncodedDataset(Dataset):

    """Dataset that loads a tokenized file and builds overlapping sliding-window sequences
    for next-token prediction.  The tail of the token stream is padded just enough to
    form one final complete window so no tokens are discarded."""
    def __init__(self, file_path, block_size, stride=None, pad_token_id=3):

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

        # Guard: if the file has fewer tokens than one window, return an empty dataset
        # (matches the original behaviour for very small or empty token files).
        if len(tokens) < window_size:
            self.sequences = torch.empty(
                (0, window_size),
                dtype=tokens.dtype,
                device=tokens.device,
            )
        else:
            # Pad the tail so no tokens are thrown away.
            # unfold() stops when there are not enough tokens left for a full window.
            # We calculate how many tokens remain after the last complete window and
            # pad just enough to turn them into one more complete window.
            N = len(tokens)
            last_start = ((N - window_size) // self.stride) * self.stride
            last_end   = last_start + window_size
            leftover   = N - last_end          # tokens not covered by any complete window

            if leftover > 0:
                next_start = last_start + self.stride
                pad_len    = next_start + window_size - N
                padding    = torch.full((pad_len,), pad_token_id, dtype=tokens.dtype)
                tokens     = torch.cat([tokens, padding])

            # Overlapping sliding windows — each token appears in multiple sequences,
            # giving the model richer contextual coverage of the corpus.
            self.sequences = tokens.unfold(0, window_size, self.stride)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = seq[:-1].clone().detach()
        target_ids = seq[1:].clone().detach()

        return {"input_ids": input_ids, "target_ids": target_ids}

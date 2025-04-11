import os 
import torch
import numpy as np
from Data_preprocessing.config import Config
import pickle
from torch.utils.data import Dataset, DataLoader

class TransformerUtils:
    @staticmethod
    def load_tokenized_data(split: str) -> list:
        """
        Load tokenized data from .pkl file.
        
        Args:
            split (str): Dataset split ("train", "valid", or "test").
        
        Returns:
            list: List of tokenized sequences.
        """
        file_path = f"{Config.TOKENIZER_DIR}/{split}_ids.pkl"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tokenized data not found at {file_path}.")
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def pad_sequences(sequences: list) -> torch.Tensor:
        """
        Dynamically pad sequences to the longest sequence in the batch (up to Config.MAX_LENGTH).
        
        Args:
            sequences (list): List of tokenized sequences.
        
        Returns:
            torch.Tensor: Padded sequences.
        """
        pad_id = Config.PAD_ID
        max_len = min(max(len(s) for s in sequences), Config.MAX_LENGTH)
        padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            seq = seq[:max_len]
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded

    @staticmethod
    def create_masks(padded: torch.Tensor) -> torch.Tensor:
        """
        Create combined causal + padding mask.
        
        Args:
            padded (torch.Tensor): Padded input sequences of shape [batch_size, seq_len].
        
        Returns:
            torch.Tensor: Combined mask of shape [batch_size, 1, seq_len, seq_len].
        """
        batch_size, seq_len = padded.size()
        
        padding_mask = (padded != Config.PAD_ID).unsqueeze(1).unsqueeze(2)  
        
        causal_mask = torch.tril(torch.ones(1, seq_len, seq_len)).unsqueeze(0)  
        
        combined_mask = padding_mask * causal_mask 
        return combined_mask

    @staticmethod
    def prepare_batch(batch: list) -> dict:
        """
        Prepare a batch for transformer training.
        
        Args:
            batch (list): List of tokenized sequences.
        
        Returns:
            dict: Dictionary containing:
                - input_ids: Tensor of shape [batch_size, seq_len - 1].
                - attention_mask: Tensor of shape [batch_size, 1, seq_len - 1, seq_len - 1].
                - labels: Tensor of shape [batch_size, seq_len - 1].
        """
        padded = TransformerUtils.pad_sequences(batch)
        mask = TransformerUtils.create_masks(padded)
        
        return {
            'input_ids': padded[:, :-1],  
            'attention_mask': mask[:, :, :-1, :-1], 
            'labels': padded[:, 1:]  
        }

    @staticmethod
    def get_dataloader(split: str) -> DataLoader:
        """
        Get a DataLoader for a specific dataset split.
        
        Args:
            split (str): Dataset split ("train", "valid", or "test").
        
        Returns:
            DataLoader: PyTorch DataLoader for the specified split.
        """
        data = TransformerUtils.load_tokenized_data(split)
        
        class PTBDataset(Dataset):
            def __len__(self):
                return len(data)
            
            def __getitem__(self, idx):
                return data[idx]
        
        return DataLoader(
            PTBDataset(),
            batch_size=Config.BATCH_SIZE,
            collate_fn=TransformerUtils.prepare_batch,
            shuffle=(split == "train")
        )

train_loader = TransformerUtils.get_dataloader("train")

batch = next(iter(train_loader))
print("Batch shapes:")
print(f"Input IDs: {batch['input_ids'].shape}")
print(f"Attention Masks: {batch['attention_mask'].shape}")
print(f"Labels: {batch['labels'].shape}")
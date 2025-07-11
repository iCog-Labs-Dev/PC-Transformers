import torch
from torch.utils.data import Dataset
from unittest.mock import patch
from Data_preprocessing.dataloader import get_loaders
import Data_preprocessing.dataloader as dataloader_mod

class DummyDataset(Dataset):
    """
    A simple dummy dataset for testing purposes.
    Returns random input and target tensors.
    """
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 10, (8,)),
            "target_ids": torch.randint(0, 10, (8,)),
        }

def test_get_loaders_runs():
    """
    Test that get_loaders runs without error and returns batches with the expected keys and shapes.
    Uses patching to mock dataset loading and DataLoader behavior for isolation.
    """
    # Patch get_datasets, Config.BATCH_SIZE, and DataLoader drop_last
    with patch('Data_preprocessing.dataloader.get_datasets', return_value=(DummyDataset(), DummyDataset(), DummyDataset())), \
         patch.object(dataloader_mod.Config, 'batch_size', 1):
        # Patch DataLoader to always use drop_last=False
        orig_DataLoader = dataloader_mod.DataLoader
        def PatchedDataLoader(*args, **kwargs):
            kwargs['drop_last'] = False
            return orig_DataLoader(*args, **kwargs)
        with patch.object(dataloader_mod, 'DataLoader', PatchedDataLoader):
            train_loader, valid_loader, test_loader = get_loaders(distributed=False)
            batch = next(iter(train_loader))
            assert "input_ids" in batch
            assert "target_ids" in batch
            assert batch["input_ids"].shape[0] > 0

def test_penn_treebank_dataset_file_not_found():
    """
    Test that PennTreebankDataset raises FileNotFoundError when the tokenized file does not exist.
    Ensures that the error message contains the missing file path and usage instructions.
    """
    from Data_preprocessing.datasets.penn_treebank import PennTreebankDataset
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_file = "nonexistent.pkl"
        try:
            _ = PennTreebankDataset(missing_file, tmpdir, block_size=8)
        except FileNotFoundError as e:
            assert missing_file in str(e)
            assert "Please tokenize the dataset first" in str(e)
        else:
            assert False, "Expected FileNotFoundError was not raised."

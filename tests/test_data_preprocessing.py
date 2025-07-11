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

def test_dummy_penn_treebank_dataset_loads():
    """
    Test that PennTreebankDataset loads dummy data from tests/dummy_data and returns correct fields.
    """
    from Data_preprocessing.datasets.penn_treebank import PennTreebankDataset
    from unittest.mock import patch
    dummy_dir = "tests/dummy_data"
    with patch("Data_preprocessing.config.Config.tokenizer_dir", dummy_dir):
        ds = PennTreebankDataset("train_ids.pkl", dummy_dir, block_size=8)
        assert len(ds) == 2
        sample = ds[0]
        assert "input_ids" in sample and "target_ids" in sample

def test_penn_treebank_dataset_filters_short_sequences(tmp_path):
    """
    Test that PennTreebankDataset filters out sequences of length <= 1.
    Only sequences with length > 1 should remain in the dataset.
    """
    from Data_preprocessing.datasets.penn_treebank import PennTreebankDataset
    import pickle
    data = [[1], [1, 2], [1, 2, 3]]
    file_path = tmp_path / "short_seqs.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    ds = PennTreebankDataset(str(file_path.name), str(tmp_path), block_size=8)
    assert len(ds) == 2


def test_penn_treebank_dataset_block_size_truncation(tmp_path):
    """
    Test that PennTreebankDataset truncates input and target sequences to the specified block_size.
    """
    from Data_preprocessing.datasets.penn_treebank import PennTreebankDataset
    import pickle
    data = [[i for i in range(20)]]  # Sequence longer than block_size
    file_path = tmp_path / "long_seq.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    block_size = 8
    ds = PennTreebankDataset(str(file_path.name), str(tmp_path), block_size=block_size)
    sample = ds[0]
    assert len(sample["input_ids"]) == block_size
    assert len(sample["target_ids"]) == block_size


def test_pad_collate_fn_pads_sequences():
    """
    Test that pad_collate_fn correctly pads input and target sequences in a batch to the same length.
    """
    from utils.model_utils import pad_collate_fn
    import torch
    batch = [
        {"input_ids": torch.tensor([1, 2]), "target_ids": torch.tensor([1, 2])},
        {"input_ids": torch.tensor([1, 2, 3]), "target_ids": torch.tensor([1, 2, 3])}
    ]
    pad_token_id = 0
    result = pad_collate_fn(batch, pad_token_id)
    assert result["input_ids"].shape == (2, 3)
    assert (result["input_ids"][0, -1] == pad_token_id)
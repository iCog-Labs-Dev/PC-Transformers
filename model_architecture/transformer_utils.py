import torch.nn.functional as F

def ids_to_one_hot(input_ids, vocab_size):
        """input_id from [B, S] to [B, S, V]"""
        return F.one_hot(input_ids, num_classes=vocab_size).float()
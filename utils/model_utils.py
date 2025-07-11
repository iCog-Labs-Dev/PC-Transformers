import os
import subprocess
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from Data_preprocessing.config import Config
from model_architecture.pc_t_model import PCTransformer
from bert_score import score as bertscore
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch, pad_token_id=0):
    input_seqs = [item["input_ids"] for item in batch]
    target_seqs = [item["target_ids"] for item in batch]

    input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=pad_token_id)
    target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=pad_token_id)

    max_len = max(
        max(seq.size(0) for seq in input_seqs),
        max(seq.size(0) for seq in target_seqs),
    )
    def _pad(seq_list):
        """Pad a list of 1-D tensors to `max_len` using `pad_token_id`."""
        return pad_sequence(
            [torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=pad_token_id) for seq in seq_list],
            batch_first=True,
            padding_value=pad_token_id,
        )
        
    input_seqs = _pad(input_seqs)
    target_seqs = _pad(target_seqs)

    return {"input_ids": input_seqs, "target_ids": target_seqs}

def load_tokenizer():
    """
    Load a pre-trained tokenizer from the specified directory in the config.
    If it doesn't exist, automatically runs the tokenizer script.

    Returns:
        Tokenizer: An instance of the loaded tokenizer.
    """
    tokenizer_path = os.path.join(Config.tokenizer_dir, "tokenizer.json")

    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at: {tokenizer_path}")
        print(" Attempting to generate tokenizer by running:")
        print("    python -m Data_preprocessing.tokenizer.bpe_tokenizer")

        try:
            subprocess.run(
                ["python", "-m", "Data_preprocessing.tokenizer.bpe_tokenizer"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Tokenizer generation failed. Please run the tokenizer script manually.") from e

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer still not found after attempting auto-generation.\n"
                f"Please run:\n  python -m Data_preprocessing.tokenizer.bpe_tokenizer"
            )

    return Tokenizer.from_file(tokenizer_path)

def load_model(model_path, config):
    """
    Load a PCTransformer model from a checkpoint file.

    Args:
        model_path (str): Path to the saved model checkpoint.
        config: Model configuration object.
    Returns:
        PCTransformer: The loaded model with weights.
    """
    model = PCTransformer(config)
    model.load_state_dict(torch.load(model_path), strict = False)
    return model

def reset_pc_modules(model):
    """
    Reset predictive coding modules in the model by clearing errors, energy, and caches.

    Args:
        model: The model containing predictive coding modules.
    """
    for module in model.modules():
        if hasattr(module, "clear_errors"):
            module.clear_errors()
        if hasattr(module, "clear_energy"):
            module.clear_energy()
        if hasattr(module, "_x_cache"):
            module._x_cache = {}
        if hasattr(module, "_embed_cache"):
            module._embed_cache = {"mu_word": None, "mu_pos": None, "step": -1}

def compute_text_metrics(predictions, targets):
    """
    Compute and print BERTScore and BLEU metrics for predicted and target texts.

    Args:
        predictions (list of str): List of generated text strings.
        targets (list of str): List of reference text strings.
    """
    print("\nComputing BERTScore and BLEU...")
    P, R, F1 = bertscore(
        predictions,
        targets,
        lang="en",
        model_type="roberta-base",
        rescale_with_baseline=True,
    )
    print(f"BERTScore (F1): {F1.mean().item():.4f}")

    smooth_fn = SmoothingFunction().method4
    tokenized_targets = [[target.split()] for target in targets]
    tokenized_pred = [pred.split() for pred in predictions]
    bleu = corpus_bleu(tokenized_targets, tokenized_pred, smoothing_function=smooth_fn)
    print(f"BLEU Score: {bleu:.4f}")

def decode_ids(tokenizer, ids, stop_at_eos = True):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    if stop_at_eos and "[EOS]" in text:
        text = text.split("[EOS]")[0].strip()
    return text

import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from Data_preprocessing.config import Config
from model_architecture.pc_t_model import PCTransformer
from bert_score import score as bertscore
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def load_tokenizer():
    """
    Load a pre-trained tokenizer from the specified directory in the config.

    Returns:
        Tokenizer: An instance of the loaded tokenizer.
    """
    tokenizer_path = os.path.join(Config.TOKENIZER_DIR, "tokenizer.json")
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
    Reset predictive coding modules in the model by clearing errors and energy.

    Args:
        model: The model containing predictive coding modules.
    """
    for module in model.modules():
        if hasattr(module, "clear_errors"):
            module.clear_errors()
        if hasattr(module, "clear_energy"):
            module.clear_energy()

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

def decode_ids(tokenizer, ids):
    """
    Decode a list of token IDs into a string using the tokenizer.

    Args:
        tokenizer: The tokenizer instance.
        ids (list of int): List of token IDs to decode.
    Returns:
        str: Decoded string.
    """
    return tokenizer.decode(ids, skip_special_tokens=True).strip()
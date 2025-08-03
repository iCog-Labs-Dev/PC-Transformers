import os
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from Data_preprocessing.config import Config
from model_architecture.pc_t_model import PCTransformer
from bert_score import score as bertscore
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.amp import autocast

def pad_collate_fn(batch, pad_token_id=0):
    input_seqs = [item["input_ids"] for item in batch]
    target_seqs = [item["target_ids"] for item in batch]

    input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=pad_token_id)
    target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=pad_token_id)

    return {"input_ids": input_seqs, "target_ids": target_seqs}

def load_tokenizer():
    tokenizer_path = os.path.join(Config.TOKENIZER_DIR, f"gpt2_tokenizer_{Config.DATASET_NAME}.json")
    tokenizer= GPT2TokenizerFast.from_pretrained(tokenizer_path)
    special_tokens = {"pad_token": "[PAD]", "eos_token": "[EOS]"}
    tokenizer.add_special_tokens(special_tokens)
    
    Config.VOCAB_SIZE = len(tokenizer) 
    Config.PAD_ID = tokenizer.pad_token_id
    Config.EOS_ID = tokenizer.eos_token_id
    return tokenizer

def load_model(model_path, config):
    model = PCTransformer(config)
    model.load_state_dict(torch.load(model_path, weights_only=True), strict = False)
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
        if hasattr(module, "_x_cache"):
            module._x_cache = {}
        if hasattr(module, "_embed_cache"):
            module._embed_cache = {"mu_word": None, "mu_pos": None, "step": -1}

def compute_text_metrics(predictions, targets):
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


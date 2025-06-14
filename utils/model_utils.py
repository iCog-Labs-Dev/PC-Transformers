import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from Data_preprocessing.config import Config
from model_architecture.pc_t_model import PCTransformer
from bert_score import score as bertscore
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def load_tokenizer():
    tokenizer_path = os.path.join(Config.TOKENIZER_DIR, "tokenizer.json")
    return Tokenizer.from_file(tokenizer_path)

def load_model(model_path, config):
    model = PCTransformer(config)
    model.load_state_dict(torch.load(model_path), strict = False)
    return model

def reset_pc_modules(model):
    for module in model.modules():
        if hasattr(module, "clear_errors"):
            module.clear_errors()
        if hasattr(module, "clear_energy"):
            module.clear_energy()

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

def decode_ids(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True).strip()
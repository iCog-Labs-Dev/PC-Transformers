import torch
from model_architecture.pc_t_model import PCTransformer
from bert_score import score as bertscore
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def load_model(model_path, config):
    model = PCTransformer(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded successfully from {model_path}")
    return model

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




import os
import time
import torch
from tokenizers import Tokenizer
from model_architecture.pc_t_model import PCTransformer
from predictive_coding.config import GPTConfig
from Data_preprocessing.dataloader import test_loader
from Data_preprocessing.config import Config
import torch.nn.functional as F
from bert_score import score as bertscore
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Helper to decode a list of token IDs into a string
def decode_ids(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True).strip()

def load_model(model_path, config):
    model = PCTransformer(config)
    model.load_state_dict(torch.load(model_path), strict = False)
    return model

def generate_text(input_ids, max_new_tokens=50, temperature=1.0):
    input_tensor = input_ids.unsqueeze(0)

    for _ in range(max_new_tokens):
        if input_tensor.size(1) > config.block_size:
            input_tensor = input_tensor[:, -config.block_size:]

        logits = model.evaluate(input_tensor)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tensor = torch.cat((input_tensor, next_token), dim=1)

    return input_tensor[0] # Return generated sequence without batch dim


# Evaluate model on test dataset with optional text metrics
def evaluate(model, dataloader, max_batches=None, compute_text_metrics=True):
    start_time = time.time()
    model.eval()
    
    total_loss = 0.0
    batch_count = 0
    
    references = []
    candidates = []
    
    if max_batches is None:
        print(f"Evaluating on the full test set...")
    else:
        print(f"Evaluating on up to {max_batches} batches...")
        
        
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            input_ids = batch["input_ids"]
            targets = batch["target_ids"]

            logits = model.evaluate(input_ids)

            # Cross-entropy loss (ignore padding index 0)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )

            total_loss += loss.item()
            batch_count += 1
            
            # Prediction accuracy calculation
            preds = torch.argmax(logits, dim=-1)
            mask = targets != 0
            correct = (preds == targets) & mask
            correct_preds += correct.sum().item()
            total_preds += mask.sum().item()
            
            if compute_text_metrics:
                for i in range(preds.size(0)):
                    pred_str = decode_ids(tokenizer, preds[i][mask[i]].tolist())
                    tgt_str = decode_ids(tokenizer, targets[i][mask[i]].tolist())
                    candidates.append(pred_str)
                    references.append(tgt_str)

                    if i == 0:
                        # Show a sample prediction
                        prompt_len = 10
                        prompt = input_ids[i][:prompt_len].cpu()
                        gen_ids = generate_text(prompt, max_new_tokens=50, temperature=0.7)
                        gen_text = decode_ids(tokenizer, gen_ids.tolist())

                        print(f"\n[Batch {batch_idx+1}, Sample {i+1}]")
                        print(f"[PROMPT ]: {decode_ids(tokenizer, prompt.tolist())}")
                        print(f"[TARGET ]: {tgt_str}")
                        print(f"[PREDICT]: {gen_text}")

    average_loss = total_loss / batch_count if batch_count > 0 else 0.0

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Cross-Entropy Loss: {average_loss:.4f}")
    
    # Compute additional text-level metrics
    if compute_text_metrics and candidates and references:
        print("\nComputing BERTScore and BLEU...")

        P, R, F1 = bertscore(
            candidates,
            references,
            lang="en",
            model_type="roberta-base",
            rescale_with_baseline=True,
        )
        print(f"BERTScore (F1): {F1.mean().item():.4f}")

        smooth_fn = SmoothingFunction().method4
        tokenized_refs = [[ref.split()] for ref in references]
        tokenized_cands = [cand.split() for cand in candidates]
        bleu = corpus_bleu(tokenized_refs, tokenized_cands, smoothing_function=smooth_fn)
        print(f"BLEU Score: {bleu:.4f}")
        
        
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.2f} seconds")
    print(f"Total Batches Processed: {batch_idx + 1}")

  

    return average_loss



tokenizer_path = os.path.join(Config.TOKENIZER_DIR, "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

config = GPTConfig(
    vocab_size = vocab_size,
    block_size=256,
    n_embed=64,
    dropout=0.1,
    local_learning_rate=1e-7,
    T=1,
    is_holding_error=True,
    num_heads=2,
    n_blocks=2,
    num_epochs=1,
    update_bias=True,
    energy_fn_name="kld"
)


model_path = "checkpoints/pc_transformer.pt"
model = load_model(model_path, config)

# Run evaluation
# Max batches can be set to limit evaluation, or None for full dataset
evaluate(model, test_loader, max_batches=10, compute_text_metrics=True)
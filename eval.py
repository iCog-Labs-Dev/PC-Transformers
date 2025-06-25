import time
import torch
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from Data_preprocessing.dataloader import test_loader
import torch.nn.functional as F
from utils.model_utils import load_tokenizer, load_model, reset_pc_modules, compute_text_metrics, decode_ids

"""Usage: python eval.py"""

def evaluate(model, dataloader, tokenizer, max_batches=None, compute_metrics=True):
    start_time = time.time()
    model.eval()
    total_energy = 0.0
    batch_count = 0
    total_ce_loss = 0.0
    pad_token_id = tokenizer.token_to_id("[PAD]")

    decoded_targets, decoded_predictions = [], []
    
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

            logits = model(targets, input_ids)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index= pad_token_id,
            )
            total_ce_loss += ce_loss.item()

            layer_energies = []
            for module in model.modules():
                if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                    energy = module.get_energy()
                    if energy is not None and not (torch.isnan(torch.tensor(energy)) if isinstance(energy, (int, float)) else False):
                        layer_energies.append(energy)

            if layer_energies:
                valid_energies = [e for e in layer_energies if not (torch.isnan(torch.tensor(e)) if isinstance(e, (int, float)) else True)]
                batch_energy = sum(valid_energies) / len(valid_energies) if valid_energies else ce_loss.item()
            else:
                batch_energy = ce_loss.item()
            total_energy += batch_energy
            batch_count += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | CE Loss: {ce_loss.item():.4f}| Batch Energy: {batch_energy:.4f}", flush=True)

        if compute_metrics:
            preds = torch.argmax(logits, dim=-1)
            mask = targets != pad_token_id
            for i in range(preds.size(0)):
                pred_str = decode_ids(tokenizer, preds[i][mask[i]].tolist(), stop_at_eos=True)
                tgt_str = decode_ids(tokenizer, targets[i][mask[i]].tolist(), stop_at_eos=True)
                decoded_predictions.append(pred_str)
                decoded_targets.append(tgt_str)
        
        reset_pc_modules(model)

    if compute_metrics and decoded_predictions and decoded_targets:
        compute_text_metrics(decoded_predictions, decoded_targets)

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
        
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.2f} seconds")
    print(f"Total Batches Processed: {batch_idx + 1}")
    print(f"Avg CE Loss: {avg_ce_loss:.4f} | Avg Energy: {avg_energy:.4f}")
    return avg_energy, avg_ce_loss 

def main():
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    config = GPTConfig(
        vocab_size = vocab_size,
        block_size=256,
        n_embed=64,
        dropout=0.1,
        local_learning_rate= 1e-5,
        T=20,
        is_holding_error=True,
        num_heads=8,
        n_blocks=4,
        num_epochs=1,
        update_bias=True,
        energy_fn_name="scaled_mse", 
        eos_token_id = tokenizer.token_to_id("[EOS]")
    )

    model_path = "checkpoints/pc_transformer.pt"
    model = load_model(model_path, config)

    # Max batches can be set to limit evaluation, or None for full dataset
    evaluate(model, test_loader, tokenizer, max_batches= None, compute_metrics=True)

if __name__ == "__main__":
    main()
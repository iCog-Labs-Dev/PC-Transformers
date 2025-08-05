import torch
import pickle
import time
import os
import pickle
from training import train
from eval import evaluate
from utils.pc_utils import cleanup_memory
from model_architecture.pc_t_model import PCTransformer
from predictive_coding.config import GPTConfig
from utils.model_utils import reset_pc_modules, load_tokenizer
from tuning.config import get_dynamic_model_config, update_global_config, normalize_energy
from tuning.dataloader import get_dynamic_batch_size, create_subset_loaders
from tuning.tuning_logs import log_trial_to_detailed_log, log_trial_to_summary
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.device_utils import setup_ddp

def broadcast_config(config_dict, device):
    """Broadcast config from rank 0 to all other ranks"""
    obj_bytes = pickle.dumps(config_dict)
    obj_tensor = torch.tensor(list(obj_bytes), dtype=torch.uint8, device=device)
    length = torch.tensor([len(obj_tensor)], device=device)

    dist.broadcast(length, src=0)
    if dist.get_rank() != 0:
        obj_tensor = torch.empty(length.item(), dtype=torch.uint8, device=device)

    dist.broadcast(obj_tensor, src=0)
    return pickle.loads(bytes(obj_tensor.tolist()))

def objective(trial, device = None, flash=False):
    """Bayesian Objective function"""
    start_time = time.time()
    model = None
    
    print(f"\nStarting Trial {trial.number}")
    
    try:
        local_rank, is_distributed = setup_ddp()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        tokenizer = load_tokenizer()
        vocab_size = len(tokenizer)

        if is_distributed and dist.is_initialized():
            if dist.get_rank() == 0:
                config = get_dynamic_model_config(trial, vocab_size, flash=flash)
                if config is None:
                    return float("inf")
                config_dict = config.__dict__
            else:
                config_dict = None

            config_dict = broadcast_config(config_dict, device)
            config = GPTConfig(**config_dict)
            update_global_config(config.__dict__)
        
        else:
            config = get_dynamic_model_config(trial, vocab_size, flash=flash)
            if config is None:
                return float("inf")
            update_global_config(config.__dict__)

        model = PCTransformer(config).to(device)  
        if is_distributed and dist.is_initialized():
            model = DDP(model, device_ids=[device.index], output_device=device.index)

        batch_size = get_dynamic_batch_size(config.n_embed, config.block_size)
        train_loader, valid_loader = create_subset_loaders(batch_size=batch_size, distributed=is_distributed and dist.is_initialized())

        if len(train_loader) == 0 or len(valid_loader) == 0:
            return float("inf")

        model.train()
        train(model, train_loader, tokenizer, config, global_step = 0, device = device)
        reset_pc_modules(model)

        model.eval()
        avg_energy, val_loss, avg_perplexity = evaluate(model, valid_loader, tokenizer, max_batches=None, device=device)
        
        normalized_energy = normalize_energy(avg_energy, config.energy_fn_name)
        combined_energy = normalized_energy + val_loss
        trial_time = (time.time() - start_time) /3600
        
        trial.set_user_attr("config", config.__dict__)
        trial.set_user_attr("ce_loss", val_loss)
        trial.set_user_attr("energy", avg_energy)
        trial.set_user_attr("normalized_energy", normalized_energy)
        trial.set_user_attr("combined_energy", combined_energy)
        trial.set_user_attr("trial_time", trial_time)

        log_trial_to_summary("tuning/bayesian_tuning_summary.txt", trial)
        log_trial_to_detailed_log("tuning/bayesian_tuning_trials.txt", trial, config, trial_time, val_loss, avg_energy, normalized_energy, combined_energy)

        return combined_energy
    
    except Exception as e:
        print("Trial failed:", e)
        trial.set_user_attr("ce_loss", "N/A")
        trial.set_user_attr("energy", "N/A")
        trial.set_user_attr("normalized_energy", "N/A")
        trial.set_user_attr("combined_energy", "N/A")
        trial.set_user_attr("trial_time", (time.time() - start_time) / 3600)

        log_trial_to_summary("tuning/bayesian_tuning_summary.txt", trial)
        return float("inf")
    
    finally:
        if model:
            reset_pc_modules(model)
            del model
        cleanup_memory()
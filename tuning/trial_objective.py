import torch
import pickle
import time
import pickle
from training import train
from eval import evaluate
from utils.pc_utils import cleanup_memory
from model_architecture.pc_t_model import PCTransformer
from predictive_coding.config import GPTConfig
from tuning.config import get_dynamic_model_config, update_global_config
from tuning.tuning_logs import log_trial_to_detailed_log, trial_batch_logger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from data_preparation.dataloader import get_loaders
from data_preparation.config import vocab_size

def combined_loss(energy, ce_loss, alpha=0.5):
    """
    Combine energy and cross-entropy loss.
    alpha: weight between energy and CE loss (0.0 = only CE, 1.0 = only energy)
    """
    return alpha * energy + (1 - alpha) * ce_loss

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

def objective(trial, device = None, flash=False, enable_batch_logging=False):
    """Bayesian Objective function"""
    start_time = time.time()
    model = None
    
    print(f"\nStarting Trial {trial.number}")
    
    try:       
        if not dist.is_initialized() or dist.get_rank() == 0:
            config = get_dynamic_model_config(trial, vocab_size, flash)
            if config is None:
                return float("inf")
            config_dict = config.__dict__
        else:
            config_dict = None

        if dist.is_initialized():
            config_dict = broadcast_config(config_dict, device)
        
        config = GPTConfig(**config_dict)
        update_global_config(config.__dict__)

        model = PCTransformer(config).to(device)  
       
        if dist.is_initialized():
            if device.type == "cuda":
                model = DDP(model, device_ids=[device.index], output_device=device.index)
            else:
                model = DDP(model)
       
        train_loader, valid_loader, _ = get_loaders(distributed=dist.is_initialized())
        
        if len(train_loader) == 0 or len(valid_loader) == 0:
            return float("inf")

        trial_logger = trial_batch_logger(trial_number=trial.number) if enable_batch_logging else None

        model.train()
        train_energy, train_perplexity, _ = train(model, train_loader, config, global_step = 0, device = device, logger=trial_logger)

        model.eval()
        avg_energy, avg_perplexity = evaluate(model, valid_loader, max_batches=None, device=device)
        
        train_ce_loss = torch.log(torch.tensor(train_perplexity)).item()
        
        alpha = 0.5
        combined_objective = combined_loss(train_energy, train_ce_loss, alpha=alpha)
        
        trial_time = (time.time() - start_time) 
        
        trial.set_user_attr("config", config.__dict__)
        trial.set_user_attr("energy", train_energy)
        trial.set_user_attr("perplexity", train_perplexity)
        trial.set_user_attr("ce_loss", train_ce_loss)
        trial.set_user_attr("combined_loss", combined_objective)
        trial.set_user_attr("alpha", alpha)
        trial.set_user_attr("trial_time", trial_time)

        trial_path = "tuning/bayesian_tuning_trials.txt"

        if not dist.is_initialized() or dist.get_rank() == 0:
            write_header = trial.number == 0 
            log_trial_to_detailed_log(trial_path, trial, config, trial_time, train_energy, write_header=write_header)

        return combined_objective
    
    except Exception as e:
        print("Trial failed:", e)
        trial.set_user_attr("energy", "N/A")
        trial.set_user_attr("perplexity", "N/A")
        trial.set_user_attr("combined_loss", "N/A")
        trial.set_user_attr("trial_time", (time.time() - start_time))

        return float("inf")
    
    finally:
        if model:
            del model
        cleanup_memory()
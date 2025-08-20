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
from tuning.config import get_dynamic_model_config, update_global_config
from tuning.dataloader import get_dynamic_batch_size, create_subset_loaders
from tuning.tuning_logs import log_trial_to_detailed_log
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.device_utils import setup_device

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
        # if "RANK" in os.environ and torch.cuda.is_available():
        #     if not dist.is_initialized():
        #         dist.init_process_group(backend="gloo")
        #     local_rank = int(os.environ["LOCAL_RANK"])
        #     device = torch.device(f"cuda:{local_rank}")
        #     torch.cuda.set_device(local_rank)
        # else:
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank, device, _ = setup_device()
        tokenizer = load_tokenizer()
        vocab_size = len(tokenizer)

        # if dist.is_initialized():
        #     if dist.get_rank() == 0:
        #         config = get_dynamic_model_config(trial, vocab_size, flash=flash)
        #         if config is None:
        #             return float("inf")
        #         config_dict = config.__dict__
        #     else:
        #         config_dict = None
        

        #     config_dict = broadcast_config(config_dict, device)
        #     config = GPTConfig(**config_dict)
        #     update_global_config(config.__dict__)
        
        # else:
        #     config = get_dynamic_model_config(trial, vocab_size, flash=flash)
        #     if config is None:
        #         return float("inf")
        #     update_global_config(config.__dict__)
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
        # if dist.is_initialized():
        #     model = DDP(model, device_ids=[device.index], output_device=device.index)
        if dist.is_initialized():
            if device.type == "cuda":
                model = DDP(model, device_ids=[device.index], output_device=device.index)
            else:
                model = DDP(model)
        batch_size = get_dynamic_batch_size(config.n_embed, config.block_size)
        train_loader, valid_loader = create_subset_loaders(batch_size=batch_size, distributed=dist.is_initialized())

        if len(train_loader) == 0 or len(valid_loader) == 0:
            return float("inf")

        model.train()
        train(model, train_loader, tokenizer, config, global_step = 0, device = device)
        reset_pc_modules(model)

        model.eval()
        avg_energy, avg_perplexity = evaluate(model, valid_loader, tokenizer, max_batches=None, device=device)
        
        trial_time = (time.time() - start_time) 
        
        trial.set_user_attr("config", config.__dict__)
        trial.set_user_attr("energy", avg_energy)
        trial.set_user_attr("trial_time", trial_time)

        trial_path = "tuning/bayesian_tuning_trials.txt"

        if not dist.is_initialized() or dist.get_rank() == 0:
            write_header = trial.number == 0 
            log_trial_to_detailed_log(trial_path, trial, config, trial_time, avg_energy, write_header=write_header)

        return avg_energy
    
    except Exception as e:
        print("Trial failed:", e)
        trial.set_user_attr("energy", "N/A")
        trial.set_user_attr("trial_time", (time.time() - start_time))

        return float("inf")
    
    finally:
        if model:
            reset_pc_modules(model)
            del model
        cleanup_memory()
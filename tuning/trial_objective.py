import torch
import pickle
import time
import pickle
import logging
from training import train
from eval import evaluate
from utils.pc_utils import cleanup_memory
from utils.model_utils import set_seed
from model_architecture.pc_t_model import PCTransformer
from predictive_coding.config import GPTConfig
from tuning.config import get_dynamic_model_config, update_global_config
from tuning.tuning_logs import log_trial_to_detailed_log, trial_batch_logger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from data_preparation.dataloader import get_loaders
from data_preparation.config import vocab_size

logger = logging.getLogger(__name__)

def combined_loss(energy):
    """Energy-only objective for tuning."""
    return energy

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
    set_seed(42 + trial.number)
    start_time = time.time()
    model = None
    
    logger.info(f"Starting Trial {trial.number}")
    
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
        # Tuning objective runs a single train+eval pass.
        config.num_epochs = 1
        update_global_config(config.__dict__)

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info("Model configuration parameters for this trial:")
            for key in [
                "vocab_size", "block_size", "peak_learning_rate", "warmup_steps", "n_embed",
                "dropout", "lr", "embed_T", "attn_T", "linear_attn_T", "fc1_T", "fc2_T", "linear_output_T",
                "num_heads", "n_blocks", "batch_size", "num_epochs", "update_bias",
                "internal_energy_fn_name", "output_energy_fn_name", "combined_internal_weight",
                "combined_output_weight", "use_flash_attention"
            ]:
                logger.info(f"{key}={getattr(config, key)}")

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

        global_step = 0
        train_energy = float("inf")
        train_perplexity = float("inf")
        avg_energy = float("inf")
        avg_perplexity = float("inf")

        model.train()
        train_energy, train_perplexity, global_step = train(
            model,
            train_loader,
            config,
            global_step=global_step,
            device=device,
            logger=trial_logger,
        )

        model.eval()
        avg_energy, avg_perplexity = evaluate(
            model,
            config,
            valid_loader,
            max_batches=None,
            device=device,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(
                f"Trial {trial.number} | "
                f"Train Energy: {train_energy:.4f} | Train Perplexity: {train_perplexity:.4f} | "
                f"Val Energy: {avg_energy:.4f} | Val Perplexity: {avg_perplexity:.4f}"
            )
        
        combined_objective = combined_loss(train_energy)
        
        trial_time = (time.time() - start_time) 
        
        trial.set_user_attr("config", config.__dict__)
        trial.set_user_attr("energy", train_energy)
        trial.set_user_attr("perplexity", train_perplexity)
        trial.set_user_attr("combined_loss", combined_objective)
        trial.set_user_attr("trial_time", trial_time)

        trial_path = "tuning/bayesian_tuning_trials.txt"

        if not dist.is_initialized() or dist.get_rank() == 0:
            write_header = trial.number == 0 
            log_trial_to_detailed_log(trial_path, trial, config, trial_time, train_energy, write_header=write_header)

        return combined_objective
    
    except Exception as e:
        logger.exception(f"Trial {trial.number} failed: {e}")
        trial.set_user_attr("energy", "N/A")
        trial.set_user_attr("perplexity", "N/A")
        trial.set_user_attr("combined_loss", "N/A")
        trial.set_user_attr("trial_time", (time.time() - start_time))

        return float("inf")
    
    finally:
        if model:
            del model
        cleanup_memory()
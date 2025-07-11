import os
import torch
import gc
import torch.distributed as dist
 
def cleanup_memory():
    """Comprehensive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def setup_device():
    if "WORLD_SIZE" in os.environ and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        ddp = True
    elif torch.cuda.is_available():
        local_rank = 0
        device = torch.device("cuda:0")
        ddp = False
    else:
        local_rank = 0
        device = torch.device("cpu")
        ddp = False
    return local_rank, device, ddp

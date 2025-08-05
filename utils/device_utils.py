import os
import torch
import torch.distributed as dist

def setup_ddp():
    """
    Initializes Distributed Data Parallel (DDP) if environment variables are set.
    Returns (local_rank, is_distributed).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank, True
    else:
        return 0, False 
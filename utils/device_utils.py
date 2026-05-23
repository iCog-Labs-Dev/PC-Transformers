import torch
from typing import List, Callable, Any, Optional
import os
import gc
import torch.distributed as dist

def create_streams_or_futures(device: torch.device, num_streams: int) -> tuple[bool, List[Any]]:
    """
    Bypass explicit streams to prevent host-device dictionary race conditions.
    PyTorch handles parallel GPU execution safely on the default stream.
    """
    return False, []

def execute_parallel(
    use_cuda: bool,
    streams_or_futures: List[Any],
    forward_fn: Callable,
    *args,
    **kwargs
) -> Optional[Any]:
    """
    Executes sequentially on the host, allowing native PyTorch async GPU execution
    while preserving strict Python dictionary updates for _x_cache.
    """
    forward_fn(*args, **kwargs)
    return None

def synchronize_execution(use_cuda: bool, streams_or_futures: List[Any]) -> None:
    """
    No explicit synchronization needed on the default stream.
    """
    pass

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
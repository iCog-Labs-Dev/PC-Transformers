import os
import torch
import torch.distributed as dist
import torch
from typing import List, Callable, Any, Optional

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


def create_streams_or_futures(device: torch.device, num_streams: int) -> tuple[bool, List[Any]]:
    """
    Creates CUDA streams or an empty futures list based on the device.

    Args:
        device (torch.device): The device to check (CPU or CUDA).
        num_streams (int): Number of streams/futures needed.

    Returns:
        tuple[bool, List[Any]]: A tuple containing:
            - use_cuda (bool): Whether to use CUDA streams.
            - streams_or_futures (List[Any]): List of CUDA streams or empty futures list.
    """
    use_cuda = torch.cuda.is_available() and device.type == 'cuda'
    if use_cuda:
        return True, [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    return False, []
  
def execute_parallel(
    use_cuda: bool,
    streams_or_futures: List[Any],
    forward_fn: Callable,
    *args,
    **kwargs
) -> Optional[Any]:
    """
    Executes a forward function in parallel using either CUDA streams or torch.jit.fork.

    Args:
        use_cuda (bool): Whether to use CUDA streams (True) or torch.jit.fork (False).
        streams_or_futures (List[Any]): List of streams (for CUDA) or futures (for CPU).
        forward_fn (Callable): The forward function to execute.
        *args: Positional arguments for the forward function.
        **kwargs: Keyword arguments for the forward function.

    Returns:
        Optional[Any]: The future object if using torch.jit.fork, None if using CUDA streams.
    """
    if use_cuda:
        stream_idx = len(streams_or_futures) - len([s for s in streams_or_futures if not s]) - 1
        with torch.cuda.stream(streams_or_futures[stream_idx]):
            forward_fn(*args, **kwargs)
        return None
    else:
        future = torch.jit.fork(forward_fn, *args, **kwargs)
        streams_or_futures.append(future)
        return future

def synchronize_execution(use_cuda: bool, streams_or_futures: List[Any]) -> None:
    """
    Synchronizes CUDA streams or waits for torch.jit.fork futures to complete.

    Args:
        use_cuda (bool): Whether CUDA streams are being used.
        streams_or_futures (List[Any]): List of streams or futures to synchronize.
    """
    if use_cuda:
        for stream in streams_or_futures:
            if stream:  # Only synchronize non-None streams
                stream.synchronize()
    else:
        for future in streams_or_futures:
            try:
                torch.jit.wait(future)
            except Exception as e:
                print(f"Error in parallel inference step: {e}")
        streams_or_futures.clear()  # Clear futures after completion

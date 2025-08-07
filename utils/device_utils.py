import torch
from typing import List, Callable, Any, Optional

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

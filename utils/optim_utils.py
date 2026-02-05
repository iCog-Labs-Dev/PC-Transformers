import torch
from typing import Literal, Callable, List, Dict, Any, Tuple

OptimizerName = Literal["sgd", "adam"]


def sgd_step(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    lr: float = 0.001,
    **kwargs
) -> None:
    """
    In-place SGD update.
    Modifies params directly — returns nothing.
    """
    for p, g in zip(params, grads):
        if g is not None:
            p.add_(-lr * g)   # ← in-place


def adam_step(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    state: List[Dict[str, torch.Tensor]],
    lr: float = 0.001,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> None:
    """
    In-place Adam update.
    Modifies params directly — returns nothing.
    """
    beta1, beta2 = betas

    for p, g, s in zip(params, grads, state):
        if g is None:
            continue

        # Get or initialize momentum & variance
        m = s.setdefault('m', torch.zeros_like(p))
        v = s.setdefault('v', torch.zeros_like(p))
        t = s.setdefault('t', 0)
        t += 1
        s['t'] = t   # update timestep

        # Bias-corrected moment estimates
        m.mul_(beta1).add_(g, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Apply update in-place
        p.add_(-lr * m_hat / (v_hat.sqrt() + eps))


def get_local_opt_step_fn(
    name: OptimizerName,
    lr: float,
    **kwargs
) -> Callable:
    """
    Returns a callable that performs the update in-place.
    Signature: fn(params: List[Tensor], grads: List[Tensor], state=None) -> None
    """
    if name == "sgd":
        return lambda p, g, s=None: sgd_step(p, g, lr=lr, **kwargs)

    if name == "adam":
        return lambda p, g, s: adam_step(p, g, s, lr=lr, **kwargs)

    raise ValueError(f"Unknown local optimizer: {name}")


def init_local_opt_state(
    name: OptimizerName,
    params: List[torch.Tensor]
) -> Any:
    if name == "sgd":
        return None

    if name == "adam":
        return [
            {
                'm': torch.zeros_like(p),
                'v': torch.zeros_like(p),
                't': 0
            }
            for p in params
        ]

    raise ValueError(f"Unknown local optimizer: {name}")
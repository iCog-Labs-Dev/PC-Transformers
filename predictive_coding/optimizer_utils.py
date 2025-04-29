from torch.optim import Optimizer

def recreate_optimizer_x(model_xs, optimizer_fn, optimizer_kwargs, manual_optimizer_fn = None)-> Optimizer:
    """Recreates the optimizer for x."""
    if manual_optimizer_fn:
        return manual_optimizer_fn()
    return optimizer_fn(list(model_xs), **optimizer_kwargs)

def recreate_optimizer_p(model_params, optimizer_fn, optimizer_kwargs, manual_optimizer_fn=None) -> Optimizer:
    """Recreates the optimizer for model parameters."""
    if manual_optimizer_fn:
        return manual_optimizer_fn()
    return optimizer_fn(list(model_params), **optimizer_kwargs)


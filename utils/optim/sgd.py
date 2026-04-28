import torch

def step_update(param, update, eta):
    """
    Runs one step of SGD over a set of parameters given updates.

    Args:
        eta: global step size to apply when adjusting parameters

    Returns:
        adjusted parameter tensor
    """
    _param = param + update * eta
    return _param


def sgd_step(opt_params, theta, updates, eta=0.001):
    """Apply SGD update to a list of tensors."""
    time_step = opt_params
    time_step = time_step + 1
    new_theta = []
    for i in range(len(theta)):
        px_i = step_update(theta[i], updates[i], eta)
        new_theta.append(px_i)
    new_opt_params = time_step
    return new_opt_params, new_theta


def sgd_init(theta):
    device = theta[0].device if len(theta) > 0 else torch.device("cpu")
    return torch.tensor(0.0, device=device)
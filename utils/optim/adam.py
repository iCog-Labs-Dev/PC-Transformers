import torch

#adam 
def step_update(param, update, g1, g2, eta, beta1, beta2, time_step, eps):
    """
    Runs one step of Adam over a set of parameters given updates.

    Args:
        param: parameter tensor to change/adjust
        update: update tensor to be applied to parameter tensor
        g1: first moment tensor
        g2: second moment tensor
        eta: global step size
        beta1: 1st moment control factor
        beta2: 2nd moment control factor
        time_step: current time step
        eps: numerical stability coefficient

    Returns:
        adjusted parameter tensor, adjusted g1, adjusted g2
    """
    _g1 = beta1 * g1 + (1.0 - beta1) * update
    _g2 = beta2 * g2 + (1.0 - beta2) * torch.square(update)
    beta1_t = torch.tensor(beta1, device=param.device, dtype=param.dtype)
    beta2_t = torch.tensor(beta2, device=param.device, dtype=param.dtype)
    g1_unb = _g1 / (1.0 - torch.pow(beta1_t, time_step).to(param.dtype))
    g2_unb = _g2 / (1.0 - torch.pow(beta2_t, time_step).to(param.dtype))
    _param = param - eta * g1_unb / (torch.sqrt(g2_unb) + eps)
    return _param, _g1, _g2


def adam_step(opt_params, theta, updates, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Apply Adam update to a list of tensors."""
    g1, g2, time_step = opt_params
    time_step = time_step + 1
    new_theta = []
    new_g1 = []
    new_g2 = []
    for i in range(len(theta)):
        px_i, g1_i, g2_i = step_update(theta[i], updates[i], g1[i], g2[i], eta, beta1, beta2, time_step, eps)
        new_theta.append(px_i)
        new_g1.append(g1_i)
        new_g2.append(g2_i)
    return (new_g1, new_g2, time_step), new_theta


def adam_init(theta):
    device = theta[0].device if len(theta) > 0 else torch.device("cpu")
    time_step = torch.tensor(0.0, device=device)
    g1 = [torch.zeros_like(theta[i]) for i in range(len(theta))]
    g2 = [torch.zeros_like(theta[i]) for i in range(len(theta))]
    return g1, g2, time_step

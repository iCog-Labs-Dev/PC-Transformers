import torch

def step_update(param, update, g1, g2, eta, beta1, beta2, time_step, eps, weight_decay):
    _g1 = beta1 * g1 + (1.0 - beta1) * update
    _g2 = beta2 * g2 + (1.0 - beta2) * torch.square(update)
    
    beta1_t = torch.tensor(beta1, device=param.device, dtype=param.dtype)
    beta2_t = torch.tensor(beta2, device=param.device, dtype=param.dtype)
    
    g1_unb = _g1 / (1.0 - torch.pow(beta1_t, time_step).to(param.dtype))
    g2_unb = _g2 / (1.0 - torch.pow(beta2_t, time_step).to(param.dtype))
    
    adam_step = eta * g1_unb / (torch.sqrt(g2_unb) + eps)

    if weight_decay > 0.0:
        _param = param + adam_step - eta * weight_decay * param
    else:
        _param = param + adam_step
    return _param, _g1, _g2


def adamw_step(opt_params, theta, updates, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    g1, g2, time_step = opt_params
    time_step = time_step + 1
    new_theta = []
    new_g1 = []
    new_g2 = []
    
    for i in range(len(theta)):
        px_i, g1_i, g2_i = step_update(
            theta[i], updates[i], g1[i], g2[i], 
            eta, beta1, beta2, time_step, eps, weight_decay
        )
        new_theta.append(px_i)
        new_g1.append(g1_i)
        new_g2.append(g2_i)
    
    return (new_g1, new_g2, time_step), new_theta


def adamw_init(theta):
    device = theta[0].device if len(theta) > 0 else torch.device("cpu")
    time_step = torch.tensor(0.0, device=device)
    g1 = [torch.zeros_like(t) for t in theta]
    g2 = [torch.zeros_like(t) for t in theta]
    return g1, g2, time_step
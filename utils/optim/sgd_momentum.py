import torch

def step_update(param, update, momentum_buffer, eta, momentum):
    v_new = momentum * momentum_buffer + update
    _param = param + eta * v_new
    return _param, v_new


def sgd_momentum_step(opt_params, theta, updates, eta=0.001, momentum=0.9):
    time_step, momentum_buffers = opt_params
    time_step = time_step + 1
    new_theta = []
    new_momentum_buffers = []
    
    for i in range(len(theta)):
        px_i, v_i = step_update(
            theta[i], 
            updates[i], 
            momentum_buffers[i], 
            eta, 
            momentum
        )
        new_theta.append(px_i)
        new_momentum_buffers.append(v_i)
    
    return (time_step, new_momentum_buffers), new_theta


def sgd_momentum_init(theta):
    device = theta[0].device if len(theta) > 0 else torch.device("cpu")
    time_step = torch.tensor(0.0, device=device)
    momentum_buffers = [torch.zeros_like(t) for t in theta]
    return time_step, momentum_buffers
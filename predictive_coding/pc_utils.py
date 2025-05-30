import torch
import torch.nn.functional as F
import math

def x_init(batch_size: int, seq_len: int, embedding_size: int) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, embedding_size)

def step_embed(t, target, layer, layer_type, input_ids, position_ids, local_lr, clamp_value, T,energy_fn_name, is_holding_error):
        word_layer = layer["word"]
        pos_layer = layer["pos"]

        mu_word = word_layer(input_ids)
        mu_pos = pos_layer(position_ids)
        mu = mu_word + mu_pos
        error = target - mu

        update = torch.clamp(error, -clamp_value, clamp_value)
        with torch.no_grad():
            for b in range(error.size(0)):
                for s in range(error.size(1)):
                    idx_w = input_ids[b, s]
                    idx_p = position_ids[b, s]
                    word_layer.weight.data[idx_w] += local_lr * update[b, s]
                    pos_layer.weight.data[idx_p] += local_lr * update[b, s]

        if t == T - 1:
            finalize_step(mu, target, error, t, layer_type,energy_fn_name, is_holding_error)

        return mu
    
def step_linear(t, target, x, layer, layer_type, local_lr, clamp_value, T, is_holding_error,energy_fn_name,update_bias = True):
        mu = layer(x)
        if layer_type == "fc1":
            mu = F.gelu(mu)

        error = target - mu
        x_update = error @ layer.weight
        delta_W = local_lr * torch.einsum("bsh,bsv->hv", error, x)

        x = torch.clamp(x + local_lr * x_update, -clamp_value, clamp_value)
        layer.weight.data.add_(delta_W)

        if layer.bias is not None and update_bias:
            layer.bias.data.add_(local_lr * error.mean(dim=(0, 1)))

        if t == T - 1:
            finalize_step(mu, target, error, t, layer_type,energy_fn_name, is_holding_error)

        return x, mu

def step_attn(t, target, x, proj_layers, layer_type, local_lr, clamp_value, T, is_holding_error,energy_fn_name, update_bias = True):
        q_proj = proj_layers["q_proj"]
        k_proj = proj_layers["k_proj"]
        v_proj = proj_layers["v_proj"]

        Q, K, V = q_proj(x), k_proj(x), v_proj(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        mask = torch.tril(torch.ones_like(scores, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = scores.softmax(dim=-1)
        mu = attn_weights @ V

        error = target - mu
        x = torch.clamp(x - local_lr * error, -clamp_value, clamp_value)

        for proj in (q_proj, k_proj, v_proj):
            delta_W = local_lr * torch.einsum("bsh,bsv->hv", error, x.detach())
            proj.weight.data.add_(delta_W)
            if proj.bias is not None and update_bias:
                proj.bias.data.add_(local_lr * error.mean(dim=(0, 1)))

        if t == T - 1:
            finalize_step(mu, target, error, t, layer_type,energy_fn_name, is_holding_error)

        return x, mu
    
ENERGY_FUNCTIONS = {
    "scaled_mse": lambda mu, x: ((mu - x) ** 2).mean(dim=-1) * 0.05,
    "mse": lambda mu, x: ((mu - x) ** 2).mean(dim=-1),
    "l1": lambda mu, x: (mu - x).abs().mean(dim=-1),
    "cosine": lambda mu, x: 1 - torch.nn.functional.cosine_similarity(mu, x, dim=-1),
    "kld": lambda mu, x: torch.nn.functional.kl_div(
        mu.log_softmax(dim=-1),
        x.softmax(dim=-1),
        reduction='mean'
    ).sum(dim=-1)
}

def energy_fn(mu: torch.Tensor, x: torch.Tensor,energy_fn_name: str) -> torch.Tensor:
    if energy_fn_name not in ENERGY_FUNCTIONS:
        raise ValueError(f"Unknown energy function: {energy_fn_name}. Choose from {list(ENERGY_FUNCTIONS.keys())}")
    return ENERGY_FUNCTIONS[energy_fn_name](mu, x)

def finalize_step(mu, target, error, t, layer_type,energy_fn_name, is_holding_error = False):
    energy = energy_fn(mu, target,energy_fn_name).mean().item() if is_holding_error else None
    errors = [{"step": t, "type": layer_type, "error": error.mean().item()}]
    return energy, errors

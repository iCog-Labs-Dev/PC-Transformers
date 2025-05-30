import torch
import torch.nn.functional as F
import math

def x_init(batch_size: int, seq_len: int, embedding_size: int) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, embedding_size)

def step_embed(t, T, target, layer, layer_type, input_ids, position_ids, local_lr, clamp_value, is_holding_error):
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
        finalize_step(mu, target, error, t, layer_type, is_holding_error)

        return mu
    
def step_linear(t, target, x, layer, W_latents, layer_type, local_lr, clamp_value, T, use_lateral, is_holding_error,update_bias = True):
        mu = layer(x)
        if layer_type == "fc1":
            mu = F.gelu(mu)

        error = target - mu
        # Latent state and W_latent update
        x_layer = error @ layer.weight

        if use_lateral and layer_type in W_latents:
            W_latent = W_latents[layer_type]
            x_latent = x @ W_latent
            x = x + local_lr * (x_layer - x_latent)

            hebbian_latent = torch.einsum("bsh,bsv->hv", x.detach(), x.detach())
            W_latents[layer_type].data.add_(local_lr * hebbian_latent)
        
        else:
            x = x + local_lr * x_layer
        
        x = torch.clamp(x, -clamp_value, clamp_value)
        
        # Hebbian Update W_layer
        delta_W = local_lr * torch.einsum("bsh,bsv->hv", error, x.detach())
        layer.weight.data.add_(delta_W)

    error = target - mu
    x_update = error @ layer.weight
    delta_W = local_lr * torch.einsum("bsh,bsv->hv", error, x)

    x = torch.clamp(x + local_lr * x_update, -clamp_value, clamp_value)
    layer.weight.data.add_(delta_W)

    if layer.bias is not None and update_bias:
        layer.bias.data.add_(local_lr * error.mean(dim=(0, 1)))

def step_attn(t, target, x, W_latents, proj_layers, layer_type, local_lr, clamp_value, T, use_lateral, is_holding_error, update_bias = True):
        assert proj_layers is not None, "proj_layers dict is required for attention"
        q_proj = proj_layers.get("q_proj", None)
        k_proj = proj_layers.get("k_proj", None)
        v_proj = proj_layers.get("v_proj", None)

        assert all(p is not None for p in (q_proj, k_proj, v_proj)), "Missing Q/K/V projections in dict"

    return x, mu

        error = target - mu
        # Latent state and W_latent update
        W_layer = (q_proj.weight.T + k_proj.weight.T + v_proj.weight.T) / 3
        x_layer = error @ W_layer

        if use_lateral and layer_type in W_latents:
            W_latent = W_latents[layer_type]
            x_latent = x @ W_latent
            x = x + local_lr * (x_layer - x_latent)

            hebbian_latent = torch.einsum("bsh,bsv->hv", x.detach(), x.detach())
            W_latents["attn"].data.add_(local_lr * hebbian_latent)
        
        else:
            x = x + local_lr * x_layer

        x = torch.clamp(x, -clamp_value, clamp_value)

        # Hebbian update W_latent
        for proj in (q_proj, k_proj, v_proj):
            delta_W = local_lr * torch.einsum("bsh,bsv->hv", error, x.detach())
            proj.weight.data.add_(delta_W)
            if proj.bias is not None and update_bias:
                proj.bias.data.add_(local_lr * error.mean(dim=(0, 1)))

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
        finalize_step(mu, target, error, t, layer_type, is_holding_error)

    return x, mu


def energy_fn(mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return ((mu - x) ** 2).mean(dim=-1) * 0.05


def finalize_step(mu, target, error, t, layer_type, is_holding_error=False):
    energy = energy_fn(mu, target).mean().item() if is_holding_error else None
    errors = [{"step": t, "type": layer_type, "error": error.mean().item()}]
    return energy, errors

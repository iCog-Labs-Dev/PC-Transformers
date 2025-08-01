import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
from predictive_coding.config import GPTConfig
from utils.attention_utils import apply_flash_attention, apply_standard_attention


def compute_DVL(attn_v, requires_update):
    B, H, T, D = attn_v.shape
    device = attn_v.device
    x = attn_v.transpose(0, 1).flatten(2, 3)
    x = F.normalize(x, p=2, dim=-1)
    s_m = torch.bmm(x, x.transpose(1, 2))
    N = s_m.size(1)
    mask = ~torch.eye(N, dtype=torch.bool, device=attn_v.device)
    s_m = s_m[:, mask].mean(dim=-1)
    identity = torch.eye(H, device=attn_v.device)
    identity = identity.unsqueeze(0).expand(H, -1, -1)
    corr = s_m - identity
    dvl = (corr ** 2).mean()
    dvl_grad = torch.zeros_like(attn_v, device=device)
    try:
        if requires_update:
            dvl_grad = torch.autograd.grad(dvl, attn_v, retain_graph=True)[0]
    except Exception as e:
        print(f"Error computing diversity gradient: {e}")
    return dvl_grad


def get_head_similarity(mu_heads):
    B, H, T, D = mu_heads.shape
    x = mu_heads.transpose(0, 1).flatten(2, 3)
    x = F.normalize(x, p=2, dim=-1)
    corr = torch.bmm(x, x.transpose(1, 2))
    mask = ~torch.eye(corr.size(1), device=corr.device).bool()
    s_v = corr[:, mask].mean(dim=-1)
    corr = s_v.abs().mean(dim=-1)
    return corr.detach().cpu()


def x_init(batch_size: int, seq_len: int, embedding_size: int, device: torch.device = None) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, embedding_size, device=device)


def step_embed(t, T, target, layer, layer_type, input_ids, position_ids, local_lr, clamp_value, energy_fn_name,
               is_holding_error, requires_update, mu_word_cache=None, mu_pos_cache=None):
    word_layer = layer["word"]
    pos_layer = layer["pos"]
    vocab_size = word_layer.weight.size(0)
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size - 1)
    max_pos = pos_layer.weight.size(0)
    if position_ids.max() >= max_pos:
        position_ids = torch.clamp(position_ids, max=max_pos - 1)

    if requires_update or mu_word_cache is None or mu_pos_cache is None:
        mu_word = word_layer(input_ids)
        mu_pos = pos_layer(position_ids)
    else:
        mu_word = mu_word_cache
        mu_pos = mu_pos_cache

    mu = mu_word + mu_pos

    if not requires_update:
        if t == T - 1:
            finalize_step(mu, target, mu - mu, t, layer_type, energy_fn_name, is_holding_error)
        return mu, mu_word, mu_pos

    error = target - mu
    update = torch.clamp(error, -clamp_value, clamp_value)

    if requires_update:
        with torch.no_grad():
            flat_input_ids = input_ids.reshape(-1)
            flat_update = update.reshape(-1, update.size(-1))
            word_weight = word_layer.weight.data.index_add(0, flat_input_ids, local_lr * flat_update)
            word_layer.weight = nn.Parameter(word_weight)

            flat_position_ids = position_ids.reshape(-1)
            pos_weight = pos_layer.weight.data.index_add(0, flat_position_ids, local_lr * flat_update)
            pos_layer.weight = nn.Parameter(pos_weight)

    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type, energy_fn_name, is_holding_error)

    return mu, mu_word, mu_pos


def step_linear(t, T, target, x, layer, W_latents, layer_type, local_lr, clamp_value, use_lateral, is_holding_error,
                energy_fn_name, update_bias, requires_update):
    device = x.device

    # Only enable gradients for output layer
    if layer_type == "linear_output":
        if not x.requires_grad:
            x.requires_grad_(True)

    mu = layer(x)
    if layer_type == "fc1":
        mu = F.gelu(mu)

    error = target - mu

    if layer_type == "linear_output":
        energy = energy_fn(mu, target, energy_fn_name="kld")

        if not torch.isfinite(energy):
            print(f"[WARNING] KLD energy is {energy} → using zero grad")
            delta_x = torch.zeros_like(x)
        else:
            try:
                grad_outputs = torch.autograd.grad(energy, x, retain_graph=True, allow_unused=True)
                delta_x = grad_outputs[0] if grad_outputs[0] is not None else torch.zeros_like(x)
            except Exception as e:
                print(f"[ERROR] autograd failed for x in {layer_type}: {e}")
                delta_x = torch.zeros_like(x)

        # Sanitize
        if delta_x is None:
            delta_x = torch.zeros_like(x)
        delta_x = torch.nan_to_num(delta_x, 0.0).clamp(-1.0, 1.0)
        x = x - local_lr * delta_x  # descent on energy

        
        if requires_update:
            try:
                delta_W = torch.autograd.grad(energy, layer.weight, retain_graph=True)[0]
                delta_b = torch.autograd.grad(energy, layer.bias, retain_graph=True)[0] if layer.bias is not None else None
            except:
                delta_W = local_lr * torch.einsum("bsh,bsv->hv", error, x.detach())
                delta_b = local_lr * error.mean(dim=(0, 1)) if layer.bias is not None else None

            layer.weight = nn.Parameter(layer.weight - local_lr * delta_W)
            if layer.bias is not None and update_bias:
                layer.bias = nn.Parameter(layer.bias - local_lr * delta_b)

    else:
        error_proj = torch.einsum("bsh, vh -> bsv", error, layer.weight.T) if layer.weight.shape[0] != layer.weight.shape[1] else error

        if use_lateral and layer_type in W_latents:
            W_latent = W_latents[layer_type].to(device)
            x_latent = torch.einsum("bsh,hv->bsv", x, W_latent)
            delta_x = error_proj + x_latent
            x = x + local_lr * delta_x

            if requires_update:
                anti_hebbian_latent = -torch.einsum("bsh,bsv->hv", x.detach(), x.detach())
                W_latents[layer_type] = W_latents[layer_type] + local_lr * anti_hebbian_latent
        else:
            x = x + local_lr * error_proj

        # pc weight update
        if requires_update:
            delta_W = local_lr * torch.einsum("bsh,bsv->hv", error, x.detach())
            layer.weight = nn.Parameter(layer.weight + delta_W)
            if layer.bias is not None and update_bias:
                layer.bias = nn.Parameter(layer.bias + local_lr * error.mean(dim=(0, 1)))

    x = torch.clamp(x, -clamp_value, clamp_value)

    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type, energy_fn_name, is_holding_error)

    return x, mu


def step_attn(t, T, target, x, W_latents, proj_layers, layer_type, local_lr, clamp_value, use_lateral, is_holding_error,
              energy_fn_name, update_bias, requires_update, layer_instance, num_heads, n_embed, la, flash=False):
    assert proj_layers is not None, "proj_layers dict is required for attention"
    device = x.device
    q_proj = proj_layers.get("q_proj", None)
    k_proj = proj_layers.get("k_proj", None)
    v_proj = proj_layers.get("v_proj", None)
    assert all(p is not None for p in (q_proj, k_proj, v_proj)), "Missing Q/K/V projections in dict"

    Q = q_proj(x)
    K = k_proj(x)
    V = v_proj(x)
    batch_size, seq_len, embed_dim = target.shape
    head_dim = n_embed // num_heads
    la = la * math.sqrt(1.0 / head_dim)

    Q = Q.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    K = K.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    V = V.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

    if flash:
        mu_heads = apply_flash_attention(Q, K, V)
    else:
        mu_heads = apply_standard_attention(Q, K, V)

    dvl_grad = compute_DVL(mu_heads, requires_update)
    if dvl_grad is not None:
        dvl_grad = dvl_grad.to(device)
    similarity = get_head_similarity(mu_heads)
    mu = mu_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    error = target - mu

    if dvl_grad is not None:
        B, T, H, D = dvl_grad.shape
        dvl_projected = dvl_grad.permute(0, 2, 1, 3).contiguous().view(B, T, -1).clamp(-1e-3, 1e-3)
        error = error + la * dvl_projected

    if layer_instance is not None:
        setattr(layer_instance, '_head_similarity', similarity)
        setattr(layer_instance, '_head_similarity_avg', similarity.mean().item())
        setattr(layer_instance, '_head_similarity_max', similarity.max().item())

    if use_lateral and layer_type in W_latents:
        W_latent = W_latents[layer_type].to(device)
        x_latent = x @ W_latent
        delta_x = error + x_latent
        x = x + local_lr * delta_x
        if requires_update:
            anti_hebbian_latent = -torch.einsum("bsh,bsv->hv", x.detach(), x.detach())
            W_latents[layer_type] = W_latent + local_lr * anti_hebbian_latent
    else:
        x = x + local_lr * error

    x = torch.clamp(x, -clamp_value, clamp_value)

    if requires_update:
        for proj in (q_proj, k_proj, v_proj):
            delta_W = local_lr * torch.einsum("bsh,bsv->hv", error, x.detach())
            proj.weight = nn.Parameter(proj.weight + delta_W)
            if proj.bias is not None and update_bias:
                proj.bias = nn.Parameter(proj.bias + local_lr * error.mean(dim=(0, 1)))

    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type, energy_fn_name, is_holding_error)

    return x, mu


# --- Energy Functions ---
ENERGY_FUNCTIONS = {
    "scaled_mse": lambda mu, x: ((mu - x) ** 2).mean(dim=-1) * 0.05,
    "mse": lambda mu, x: ((mu - x) ** 2).mean(dim=-1),
    "l1": lambda mu, x: (mu - x).abs().mean(dim=-1),
    "cosine": lambda mu, x: 1 - F.cosine_similarity(mu, x, dim=-1),
    "kld": lambda mu, target: torch.clamp(
        F.kl_div(
            F.log_softmax(mu, dim=-1).clamp(min=-50),
            target,  # ← must be one-hot or soft labels
            reduction='batchmean'
        ),
        min=0.0,
        max=100.0
    )
}


def energy_fn(mu: torch.Tensor, x: torch.Tensor, energy_fn_name: str) -> torch.Tensor:
    if energy_fn_name not in ENERGY_FUNCTIONS:
        raise ValueError(f"Unknown energy function: {energy_fn_name}. Choose from {list(ENERGY_FUNCTIONS.keys())}")
    return ENERGY_FUNCTIONS[energy_fn_name](mu, x)


def finalize_step(mu, target, error, t, layer_type, energy_fn_name, is_holding_error=False):
    device = mu.device
    target = target.to(device)
    error = error.to(device)
    energy = energy_fn(mu, target, energy_fn_name).mean().item() if is_holding_error else None
    errors = [{"step": t, "type": layer_type, "error": error.mean().item()}]
    return energy, errors


def ids_to_one_hot(input_ids, vocab_size):
    device = input_ids.device
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size - 1)
    return F.one_hot(input_ids, num_classes=vocab_size).float().to(device)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
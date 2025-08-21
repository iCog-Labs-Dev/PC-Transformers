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
    x = attn_v.transpose(0, 1).flatten(2, 3)  # (H, B, T*D)
    x = x.transpose(0, 1)  
    x = F.normalize(x, p=2, dim=-1)
    s_m = torch.bmm(x, x.transpose(1, 2))  
    s_m = s_m.mean(dim=0)  
    identity = torch.eye(H, device=attn_v.device)  
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
    x = mu_heads.transpose(0, 1).flatten(2, 3)  # [H, N, D]
    x = F.normalize(x, p=2, dim=-1)
    corr = torch.bmm(x, x.transpose(1, 2))  
    mask = ~torch.eye(corr.size(1), device=corr.device).bool()
    s_v = corr[:, mask].mean(dim= -1)
    corr = s_v.abs().mean(dim=-1)  

    return corr.detach().cpu()
    
def x_init(batch_size: int, seq_len: int, embedding_size: int, device: torch.device = None) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, embedding_size, device = device)
def step_embed(t, T, target, layer, layer_type, input_ids, position_ids, local_lr, clamp_value, energy_fn_name, is_holding_error, requires_update, layer_norm, mu_word_cache=None, mu_pos_cache=None):
    """
    Perform a predictive coding update step for the embedding layer.
    Now supports vectorized updates and caching of mu_word/mu_pos for inference.
    Args:
        t (int): Current inference step.
        T (int): Total number of inference steps.
        target (torch.Tensor): Target activity tensor.
        layer (dict): Dictionary with 'word' and 'pos' embedding layers.
        layer_type (str): Layer type string.
        input_ids (torch.Tensor): Input token IDs.
        position_ids (torch.Tensor): Position IDs.
        local_lr (float): Local learning rate.
        clamp_value (float): Value to clamp updates.
        energy_fn_name (str): Name of energy function.
        is_holding_error (bool): Whether to accumulate errors.
        requires_update (bool): Whether to update weights.
        mu_word_cache, mu_pos_cache: Optional cached values for inference.
    Returns:
        tuple: (mu, mu_word, mu_pos)
    """
    word_layer = layer["word"]
    pos_layer = layer["pos"]

    # Clip input_ids and position_ids to valid ranges
    vocab_size = word_layer.weight.size(0)
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size-1)
        
    max_pos = pos_layer.weight.size(0)
    if position_ids.max() >= max_pos:
        position_ids = torch.clamp(position_ids, max=max_pos-1)

    if requires_update or mu_word_cache is None or mu_pos_cache is None:
        mu_word = word_layer(input_ids)
        mu_word = torch.clamp(mu_word, -10.0, 10.0)
        mu_pos = pos_layer(position_ids)
        mu_pos = torch.clamp(mu_pos, -10.0, 10.0)
    else:
        mu_word = mu_word_cache
        mu_pos = mu_pos_cache
    mu = mu_word + mu_pos
    mu_norm=layer_norm(mu)
    
    error = target - mu_norm
    if not requires_update:
        if t == T - 1:
            finalize_step(mu, target, mu - mu, t, layer_type, energy_fn_name, is_holding_error)
        return mu, mu_word, mu_pos, error
 
    update = torch.clamp(error, -clamp_value, clamp_value)
    if requires_update: 
        with torch.no_grad():
            flat_input_ids = input_ids.reshape(-1)
            flat_update = update.reshape(-1, update.size(-1))

            flat_position_ids = position_ids.reshape(-1)
            delta = local_lr * flat_update
            delta = torch.clamp(delta, -0.01, 0.01)
            word_layer.weight.data.index_add_(0, flat_input_ids, delta)
            pos_layer.weight.data.index_add_(0, flat_position_ids, delta)
            
    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type, energy_fn_name, is_holding_error)
    # print(f"mu_word: max={mu_word.abs().max():.3f}, nan={torch.isnan(mu_word).any()}")
    # print(f"mu_pos: max={mu_pos.abs().max():.3f}, nan={torch.isnan(mu_pos).any()}")
    # print(f"embedding weights: max={word_layer.weight.abs().max():.3f}")
    
    return mu, mu_word, mu_pos, error
    
def step_linear(t, T, target, x, layer, W_latents, layer_type, local_lr, clamp_value, use_lateral, is_holding_error, energy_fn_name, update_bias, requires_update, td_err, layer_norm):
    """
    Perform a predictive coding update step for a linear (fully connected) layer.

    Args:
        t (int): Current inference step.
        T (int): Total number of inference steps.
        target (torch.Tensor): Target activity tensor.
        x (torch.Tensor): Current activity tensor.
        layer (nn.Module): Linear layer.
        W_latents (dict): Lateral weights.
        layer_type (str): Layer type string.
        local_lr (float): Local learning rate.
        clamp_value (float): Value to clamp updates.
        use_lateral (bool): Whether to use lateral connections.
        is_holding_error (bool): Whether to accumulate errors.
        energy_fn_name (str): Name of energy function.
        update_bias (bool): Whether to update bias.
        requires_update (bool): Whether to update weights.
    Returns:
        tuple: (updated activity tensor, predicted output tensor)
    """
    device = x.device
    #print(f"Initial x: mean={x.mean():.3f}, std={x.std():.3f}, nan={torch.isnan(x).any()}")
    if layer_norm is not None and layer_type == "fc1":
        x = layer_norm(x)
    elif layer_type=="fc2":
        x=F.gelu(x)
    else:
        x = x
        
    mu = layer(x)
    if layer_type == "fc1":
        mu = F.gelu(mu)
        
    if layer_norm is not None and layer_type in ["linear_attn", "fc2"]:
       mu = layer_norm(mu)
        
        
    mu = torch.clamp(mu, -10.0, 10.0)
    
    bu_err = target - mu      
    error_proj = bu_err @ layer.weight    
    if td_err is not None:
        error= error_proj - td_err
    else:
        error= error_proj

    if use_lateral and layer_type in W_latents:
        W_latent = W_latents[layer_type].to(device) 
        x_latent = torch.einsum("bsh,hv->bsv", x, W_latent)
        delta_x = error + x_latent
        x = x + local_lr * delta_x
        if requires_update:
            anti_hebbian_latent = -torch.einsum("bsh,bsv->hv", x.detach(), x.detach())
            W_latents[layer_type] = W_latents[layer_type] + local_lr * anti_hebbian_latent
            W_latents[layer_type].data = F.normalize(W_latents[layer_type].data, p=2, dim=1)
    
    else:
        x= x + local_lr * error 
    
    x = torch.clamp(x, clamp_value, clamp_value)
    
    # PC Update W_layer
    if requires_update:
        delta_W = local_lr * torch.einsum("bsv, bsh -> vh", bu_err, x.detach())
        delta_W = torch.clamp(delta_W, -0.01, 0.01)
        layer.weight.data.add_(delta_W)
        if layer.bias is not None and update_bias:
            delta_b = local_lr * bu_err.mean(dim=(0, 1))
            delta_b = torch.clamp(delta_b, -0.01, 0.01)
            layer.bias.data.add_(delta_b)

    x = torch.clamp(x, -clamp_value, clamp_value)
    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type,energy_fn_name, is_holding_error)
    # print(f"x: mean={x.mean():.3f}, std={x.std():.3f}, nan={torch.isnan(x).any()}")
    # print(f"layer weights: max={layer.weight.abs().max():.3f}, nan={torch.isnan(layer.weight).any()}")
    # print(f"bias: max={layer.bias.abs().max() if layer.bias is not None else 'N/A'}")

    return x, mu, bu_err

def step_attn(t, T, target, x, W_latents, proj_layers, layer_type, local_lr, clamp_value, use_lateral, is_holding_error, energy_fn_name, update_bias, requires_update, layer_instance, num_heads, n_embed, la, td_err,layer_norm, flash=False):
        assert proj_layers is not None, "proj_layers dict is required for attention"
        device = x.device
        x=layer_norm(x)
        q_proj = proj_layers.get("q_proj", None)
        k_proj = proj_layers.get("k_proj", None)
        v_proj = proj_layers.get("v_proj", None)
        
        assert all(p is not None for p in (q_proj, k_proj, v_proj)), "Missing Q/K/V projections in dict"        
        Q= q_proj(x)
        K= k_proj(x)
        V= v_proj(x)
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
        dvl_norm = dvl_grad.norm().item() if dvl_grad is not None else 0.0
        similarity = get_head_similarity(mu_heads)
        mu = mu_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        mu = torch.clamp(mu, -10.0, 10.0)
        
        if energy_fn_name=="kld":
            energy = energy_fn(mu, target, energy_fn_name)
            if not torch.isfinite(energy).all():
                 delta_x = torch.zeros_like(x)
            else:
                try:
                    grad_qkv = torch.autograd.grad(energy, x, retain_graph=True, allow_unused=True)
                    delta_x = grad_qkv[0] if grad_qkv[0] is not None else torch.zeros_like(x)
                    #delta_x= delta_x - td_err
                except Exception as e:
                    print(f"[ERROR] autograd failed for x in {layer_type}: {e}")
                    delta_x = torch.zeros_like(x)

            if delta_x is None:
               delta_x = torch.zeros_like(x)
               delta_x = torch.nan_to_num(delta_x, 0.0).clamp(-1.0, 1.0)

            x = x + local_lr * delta_x 
            x = torch.clamp(x, -clamp_value, clamp_value)

            if requires_update:
                
                for proj in (q_proj, k_proj, v_proj):
                      grad_weight = torch.autograd.grad(energy, proj.weight, retain_graph=True, allow_unused=True)[0]
                      delta_W = local_lr * grad_weight
                      proj.weight.data.add_(delta_W)
                      if proj.bias is not None and update_bias:
                        grad_bias = torch.autograd.grad(energy, proj.bias, retain_graph=True, allow_unused=True)[0]
                        delta_b = (local_lr * grad_bias).clamp(-0.01, 0.01)
                        delta_b = delta_b.view(-1)
                        proj.bias.data.add_(delta_b)
                   
        
        else:
            bu_err = target - mu  # B, T, D
            if td_err is not None:
               error= bu_err - td_err
            else:
                error = bu_err  
            
        
            if dvl_grad is not None:
                B, T, H, D = dvl_grad.shape
                dvl_projected = dvl_grad.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
                dvl_projected=dvl_projected.clamp(-1e-3, 1e-3)
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
                   anti_hebbian_latent = - torch.einsum("bsh,bsv->hv", x.detach(), x.detach())
                   W_latents[layer_type] =W_latent + local_lr * anti_hebbian_latent
                   W_latents[layer_type].data = F.normalize(W_latents[layer_type].data, p=2, dim=1)
                else:
                   x= x+ local_lr * error

            x = torch.clamp(x, -clamp_value, clamp_value)

        # PC update W_latent
            if requires_update:
               for proj in (q_proj, k_proj, v_proj):
                   delta_W = local_lr * torch.einsum("bsv, bsh -> vh", bu_err, x.detach())
                   delta_W = torch.clamp(delta_W, -0.01, 0.01)
                   proj.weight.data.add_(delta_W)
                   if proj.bias is not None and update_bias:
                      delta_b = local_lr * bu_err.mean(dim=(0, 1))
                      delta_b = torch.clamp(delta_b, -0.01, 0.01)
                      delta_b = delta_b.view(-1)
                      proj.bias.data.add_(delta_b)
 
        if t == T - 1:
            finalize_step(mu, target, error, t, layer_type,energy_fn_name, is_holding_error)
        # print(f"mu_heads: max={mu_heads.abs().max():.3f}, nan={torch.isnan(mu_heads).any()}")
        # print(f"DVL grad norm={dvl_grad.norm().item() if dvl_grad is not None else 0.0}")
        # print(f"attention weights max={q_proj.weight.abs().max():.3f}, nan={torch.isnan(q_proj.weight).any()}")
        if energy_fn_name!="kld":
           return x, mu, bu_err
        else:
            return x, mu, energy
    
ENERGY_FUNCTIONS = {
    "scaled_mse": lambda mu, x: ((mu - x) ** 2).mean(dim=-1) * 0.05,
    "mse": lambda mu, x: ((mu - x) ** 2).mean(dim=-1),
    "pc_e": lambda mu, x: ((mu - x) ** 2).mean(dim=-1)*0.5,    
    "l1": lambda mu, x: (mu - x).abs().mean(dim=-1),
    "cosine": lambda mu, x: 1 - F.cosine_similarity(mu, x, dim=-1),
    "kld": lambda mu, x: F.kl_div(
    F.log_softmax(mu / 1.0, dim=-1),
    F.softmax(x / 1.0, dim=-1),
    reduction='batchmean'
)
}

def energy_fn(mu: torch.Tensor, x: torch.Tensor,energy_fn_name: str) -> torch.Tensor:
    """
    Compute the energy (error) between predicted and target activity using the specified function.

    Args:
        mu (torch.Tensor): Predicted activity.
        x (torch.Tensor): Target activity.
        energy_fn_name (str): Name of energy function ('scaled_mse', 'mse', 'l1', 'cosine', 'kld').
    Returns:
        torch.Tensor: Computed energy value.
    """
    if energy_fn_name not in ENERGY_FUNCTIONS:
        raise ValueError(f"Unknown energy function: {energy_fn_name}. Choose from {list(ENERGY_FUNCTIONS.keys())}")
    return ENERGY_FUNCTIONS[energy_fn_name](mu, x)

def finalize_step(mu, target, error, t, layer_type,energy_fn_name, is_holding_error = False):
    """
    Finalize a predictive coding inference step by computing energy and error statistics.

    Args:
        mu (torch.Tensor): Predicted activity.
        target (torch.Tensor): Target activity.
        error (torch.Tensor): Error tensor.
        t (int): Current inference step.
        layer_type (str): Layer type string.
        energy_fn_name (str): Name of energy function.
        is_holding_error (bool): Whether to accumulate errors.
    Returns:
        tuple: (energy value, list of error statistics)
    """
    device = mu.device
    target = target.to(device)
    error = error.to(device)
    energy = energy_fn(mu, target,energy_fn_name).mean().item() if is_holding_error else None
    errors = [{"step": t, "type": layer_type, "error": error.mean().item()}]
    return energy, errors
    
def ids_to_one_hot(input_ids, vocab_size):
    """
    Convert input token IDs to one-hot encoded tensor.

    Args:
        input_ids (torch.Tensor): Tensor of shape (B, S) with token IDs.
        vocab_size (int): Size of the vocabulary.
    Returns:
        torch.Tensor: One-hot encoded tensor of shape (B, S, vocab_size).
    """
    device = input_ids.device

    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size-1)
    
    return F.one_hot(input_ids, num_classes=vocab_size).float().to(device)

def cleanup_memory():
    """Comprehensive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
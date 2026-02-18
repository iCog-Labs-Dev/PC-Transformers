import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from typing import Optional, Tuple, Any
from utils.attention_utils import apply_flash_attention, apply_standard_attention
from utils.optim.optim_utils import PCOptimizer
    
def x_init(batch_size: int, seq_len: int, embedding_size: int, device: torch.device = None) -> torch.Tensor:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return torch.zeros(batch_size, seq_len, embedding_size, device=device)


@torch.no_grad()
def step_q_embed(
    layer: dict,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Projection-only embedding step (no updates). Returns (q_embed, q_word, q_pos)."""
    word_layer: nn.Embedding = layer["word"]
    pos_layer: nn.Embedding = layer["pos"]

    vocab_size = word_layer.weight.size(0)
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size - 1)

    max_pos = pos_layer.weight.size(0)
    if position_ids.max() >= max_pos:
        position_ids = torch.clamp(position_ids, max=max_pos - 1)

    q_word = word_layer(input_ids)
    q_pos = pos_layer(position_ids)
    q_embed = q_word + q_pos
    return q_embed, q_word, q_pos


@torch.no_grad()
def step_q_attn(
    x: torch.Tensor,
    proj_layers: dict,
    num_heads: int,
    n_embed: int,
    layer_norm: Optional[nn.Module] = None,
    flash: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Projection-only attention step (no updates). Returns (q_attn, attn_context)."""
    assert proj_layers is not None, "proj_layers dict is required for q attention"

    x_norm = layer_norm(x) if layer_norm is not None else x
    q_proj = proj_layers["q_proj"]
    k_proj = proj_layers["k_proj"]
    v_proj = proj_layers["v_proj"]

    batch_size, seq_len, embed_dim = x.shape
    head_dim = n_embed // num_heads

    Q = q_proj(x_norm).view(batch_size, num_heads, seq_len, head_dim)
    K = k_proj(x_norm).view(batch_size, num_heads, seq_len, head_dim)
    V = v_proj(x_norm).view(batch_size, num_heads, seq_len, head_dim)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)

    if flash:
        mu_heads = apply_flash_attention(Q, K, V, mask=causal_mask)
    else:
        mu_heads = apply_standard_attention(Q, K, V, mask=causal_mask)

    attn_context = mu_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    q_attn = x
    return q_attn, attn_context


@torch.no_grad()
def step_q_linear(
    x: torch.Tensor,
    layer: nn.Module,
    layer_type: str,
    layer_norm: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Projection-only linear step (no updates). Returns (q_linear_input, projected_output)."""
    if layer_norm is not None and layer_type == "fc1":
        x_input = layer_norm(x)
    elif layer_type == "fc2":
        x_input = F.gelu(x)
    else:
        x_input = x

    mu = layer(x_input)

    if layer_type == "fc1":
        mu = F.gelu(mu)
    elif layer_norm is not None and layer_type in ["linear_attn", "fc2"]:
        mu = layer_norm(mu)

    q_linear_input = x
    return q_linear_input, mu

def step_embed(
    t: int,
    T: int,
    target: torch.Tensor,
    layer: dict,
    layer_type: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    requires_update: bool,
    layer_norm: Optional[nn.Module] = None,
    optimizer: Optional[PCOptimizer] = None,
    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for embedding layer.
    Returns (mu, mu_word, mu_pos, error)
    """
    word_layer: nn.Embedding = layer["word"]
    pos_layer: nn.Embedding = layer["pos"]
    
    # clip ids
    vocab_size = word_layer.weight.size(0)
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size-1)
    max_pos = pos_layer.weight.size(0)
    if position_ids.max() >= max_pos:
        position_ids = torch.clamp(position_ids, max=max_pos-1)
         
    mu_word = word_layer(input_ids)
    mu_pos = pos_layer(position_ids)
        
    mu = mu_word + mu_pos
    mu_norm=layer_norm(mu) if layer_norm is not None else mu

    error = target - mu_norm
        
    if requires_update:
        with torch.no_grad():
            flat_input_ids = input_ids.reshape(-1)
            flat_update = error.reshape(-1, error.size(-1))
            flat_position_ids = position_ids.reshape(-1)

            if optimizer is not None:
                update_word = torch.zeros_like(word_layer.weight)
                update_pos = torch.zeros_like(pos_layer.weight)
                update_word.index_add_(0, flat_input_ids, flat_update)
                update_pos.index_add_(0, flat_position_ids, flat_update)

                optimizer.step_param(word_layer.weight, update_word, local_lr, clamp_value=0.01)
                optimizer.step_param(pos_layer.weight, update_pos, local_lr, clamp_value=0.01)
            else:
                delta = local_lr * flat_update
                delta = torch.clamp(delta, -0.01, 0.01)

                word_layer.weight.data.index_add_(0, flat_input_ids, delta)
                pos_layer.weight.data.index_add_(0, flat_position_ids, delta)
            
    if t == T - 1:
           finalize_step(mu, target, error, t, layer_type, energy_fn_name)
  
    return mu, mu_word, mu_pos, error
    
def step_linear(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    layer: nn.Module,
    lateral_conn: Optional[Any], 
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module], 
    optimizer: Optional[PCOptimizer] = None,
   ):
    """
    Predictive coding step for linear-like layers.
    Returns: (updated_x, mu, bu_err)
    """
    if layer_norm is not None and layer_type == "fc1":
        x_input = layer_norm(x)
    elif layer_type == "fc2":
        x_input = F.gelu(x)
    else:
        x_input = x
        
    mu = layer(x_input)
        
    if layer_type == "fc1":
        mu = F.gelu(mu)
    elif layer_norm is not None and layer_type in ["linear_attn", "fc2"]:
        mu = layer_norm(mu)
            
    if layer_type=="linear_output":
        bu_err= target - F.softmax(mu, dim=-1) 
    else:    
        bu_err = target - mu 
        
    # project bottom-up error through weights
    error_proj= bu_err @ layer.weight      
    error = error_proj- td_err if td_err is not None else error_proj  
    
    if lateral_conn is not None:
        delta_x = lateral_conn.forward(x, error)
        x = x + local_lr * delta_x

        if requires_update:
            lateral_conn.update_weights(x.detach(), optimizer=optimizer, clamp_value=0.01)
    else:
        x= x + local_lr * error 

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))
    
    # parameter updates for the layer
    if requires_update:
        B, S = bu_err.shape[:2]
        scale = max(B * S, 1)
        update_w = torch.einsum("bsv, bsh -> vh", bu_err, x_input.detach()) / scale
        if optimizer is not None:
            optimizer.step_param(layer.weight, update_w, local_lr, clamp_value=0.01)
        else:
            delta_W = torch.clamp(local_lr * update_w, -0.01, 0.01)
            layer.weight.data.add_(delta_W)

        if layer.bias is not None and update_bias:
            update_b = bu_err.mean(dim=(0, 1))
            if optimizer is not None:
                optimizer.step_param(layer.bias, update_b, local_lr, clamp_value=0.01)
            else:
                delta_b = torch.clamp(local_lr * update_b, -0.01, 0.01)
                layer.bias.data.add_(delta_b)

    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type,energy_fn_name)

    return x, mu, bu_err

def step_attn(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    lateral_conn: Optional[Any],
    proj_layers: dict,
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    num_heads: int,
    n_embed: int,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
    flash: bool = False,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    optimizer: Optional[PCOptimizer] = None,
    ):
    """
    Predictive coding step for attention with KV caching support.
    Returns (updated_x, mu, bu_err).
    - proj_layers must contain 'q_proj','k_proj','v_proj' modules
    """
    assert proj_layers is not None, "proj_layers dict is required for attention"

    device = x.device
    
    x_norm=layer_norm(x) if layer_norm is not None else x
        
    q_proj = proj_layers["q_proj"]
    k_proj = proj_layers["k_proj"]
    v_proj = proj_layers["v_proj"]
    assert q_proj is not None and k_proj is not None and v_proj is not None, "Missing Q/K/V projections"  
        
    batch_size, seq_len, embed_dim = target.shape
    head_dim = n_embed // num_heads
   
    Q= q_proj(x_norm)
    
    # KV Cache logic: only compute K,V for new tokens if cache exists
    if use_cache and kv_cache is not None:
        K_new = k_proj(x_norm)
        V_new = v_proj(x_norm)
        
        K_cached, V_cached = kv_cache
        K = torch.cat([K_cached, K_new], dim=1)
        V = torch.cat([V_cached, V_new], dim=1)
    else:
        # Compute full K, V
        K = k_proj(x_norm)
        V = v_proj(x_norm)
    
    new_kv_cache = (K.detach(), V.detach()) if use_cache else None
    Q = Q.view(batch_size, num_heads, seq_len, head_dim)
    K = K.view(batch_size, num_heads, -1, head_dim)
    V = V.view(batch_size, num_heads, -1, head_dim)
        
    #create causal mask (1=keep, 0=mask)
    kv_len = K.size(2)
    causal_mask = torch.tril(torch.ones(seq_len, kv_len, device=device)).unsqueeze(0).unsqueeze(0)

    # !! Causal Mask
    if flash:
        mu_heads = apply_flash_attention(Q, K, V, mask=causal_mask)
    else:
        mu_heads = apply_standard_attention(Q, K, V, mask=causal_mask)
    
    mu = mu_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    
    bu_err = target - mu  # B, T, D
    error = bu_err - td_err if td_err is not None else bu_err  
                
    if lateral_conn is not None:
        delta_x = lateral_conn.forward(x, error)
        x = x + local_lr * delta_x
        
        if requires_update:
            lateral_conn.update_weights(x.detach(), optimizer=optimizer, clamp_value=0.01)
    else:
        x = x + local_lr * error

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    # PC update W_latent
    if requires_update:
        with torch.no_grad():
            B, S = batch_size, seq_len

            scale = max(B * S, 1)
            update_q = torch.einsum("bsv,bse->ve", bu_err, x_norm.detach()) / scale
            update_k = torch.einsum("bsv,bse->ve", bu_err, x_norm.detach()) / scale
            update_v = torch.einsum("bsv,bse->ve", bu_err, x_norm.detach()) / scale

            update_b_q = bu_err.mean(dim=(0, 1)) if (update_bias and q_proj.bias is not None) else None
            update_b_k = bu_err.mean(dim=(0, 1)) if (update_bias and k_proj.bias is not None) else None
            update_b_v = bu_err.mean(dim=(0, 1)) if (update_bias and v_proj.bias is not None) else None

            if optimizer is not None:
                optimizer.step_param(q_proj.weight, update_q, local_lr, clamp_value=0.01)
                optimizer.step_param(k_proj.weight, update_k, local_lr, clamp_value=0.01)
                optimizer.step_param(v_proj.weight, update_v, local_lr, clamp_value=0.01)

                if update_bias:
                    if update_b_q is not None:
                        optimizer.step_param(q_proj.bias, update_b_q, local_lr, clamp_value=0.01)
                    if update_b_k is not None:
                        optimizer.step_param(k_proj.bias, update_b_k, local_lr, clamp_value=0.01)
                    if update_b_v is not None:
                        optimizer.step_param(v_proj.bias, update_b_v, local_lr, clamp_value=0.01)
            else:
                q_proj.weight.data.add_(torch.clamp(local_lr * update_q, -0.01, 0.01))
                k_proj.weight.data.add_(torch.clamp(local_lr * update_k, -0.01, 0.01))
                v_proj.weight.data.add_(torch.clamp(local_lr * update_v, -0.01, 0.01))

                if update_bias:
                    if update_b_q is not None:
                        q_proj.bias.data.add_(torch.clamp(local_lr * update_b_q, -0.01, 0.01))
                    if update_b_k is not None:
                        k_proj.bias.data.add_(torch.clamp(local_lr * update_b_k, -0.01, 0.01))
                    if update_b_v is not None:
                        v_proj.bias.data.add_(torch.clamp(local_lr * update_b_v, -0.01, 0.01))
 
    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type,energy_fn_name)
     
    return x, mu, bu_err, new_kv_cache
    
ENERGY_FUNCTIONS = {
    "pc_e": lambda mu, x: ((mu - x) ** 2) * 0.5,    
    "kld": lambda mu, x: torch.clamp(
        F.kl_div(mu.log_softmax(dim=-1), x, reduction="batchmean"), min=0.0, max=100.0
    ),
}

def energy_fn(mu: torch.Tensor, x: torch.Tensor,energy_fn_name: str) -> torch.Tensor:
    if energy_fn_name not in ENERGY_FUNCTIONS:
        raise ValueError(f"Unknown energy function: {energy_fn_name}. Choose from {list(ENERGY_FUNCTIONS.keys())}")
    return ENERGY_FUNCTIONS[energy_fn_name](mu, x)

def finalize_step(mu: torch.Tensor, target: torch.Tensor, error: torch.Tensor, t: int, layer_type: str, energy_fn_name: str):
    device = mu.device
    target = target.to(device)
    error = error.to(device)
    energy = float(energy_fn(mu, target, energy_fn_name).mean().item())
    errors = [{"step": t, "type": layer_type, "error": error.mean().item()}]
    return energy, errors
    
def ids_to_one_hot(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
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
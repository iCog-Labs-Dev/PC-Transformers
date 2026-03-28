import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from typing import Optional, Tuple, Any, Dict
from utils.attention_utils import apply_flash_attention, apply_standard_attention
    
def x_init(batch_size: int, seq_len: int, embedding_size: int, device: torch.device = None) -> torch.Tensor:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return torch.randn(batch_size, seq_len, embedding_size, device = device)

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
    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
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
        
    update_state = None
    if requires_update:
        update_state = {
            "kind": "embed",
            "layer": layer,
            "input_ids": input_ids.detach(),
            "position_ids": position_ids.detach(),
            "error": error.detach(),
        }

    return mu, mu_word, mu_pos, error, update_state
    
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
   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
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
    
    lateral_x = None
    if lateral_conn is not None:
        delta_x = lateral_conn.forward(x, error)
        x = x + local_lr * delta_x

        if requires_update:
            lateral_x = x.detach()
    else:
        x= x + local_lr * error

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))
    
    update_state = None
    if requires_update:
        update_state = {
            "kind": "linear",
            "layer": layer,
            "bu_err": bu_err.detach(),
            "x_input": x_input.detach(),
            "update_bias": update_bias,
            "lateral_conn": lateral_conn,
            "lateral_x": lateral_x,
        }

    return x, mu, bu_err, update_state

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Dict[str, Any]]]:
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
                
    lateral_x = None
    if lateral_conn is not None:
        delta_x = lateral_conn.forward(x, error)
        x = x + local_lr * delta_x
        
        if requires_update:
            lateral_x = x.detach()
    else:
        x = x + local_lr * error

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    K_update = K[:, :, -seq_len:, :]
    V_update = V[:, :, -seq_len:, :]

    update_state = None
    if requires_update:
        update_state = {
            "kind": "attn",
            "proj_layers": proj_layers,
            "q": Q.detach(),
            "k": K_update.detach(),
            "v": V_update.detach(),
            "x_norm": x_norm.detach(),
            "update_bias": update_bias,
            "lateral_conn": lateral_conn,
            "lateral_x": lateral_x,
        }

    return x, mu, bu_err, new_kv_cache, update_state
    
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

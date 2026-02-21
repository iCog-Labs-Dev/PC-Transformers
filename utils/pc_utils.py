import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from typing import Optional, Tuple, Any
from utils.attention_utils import apply_flash_attention, apply_standard_attention
    
def x_init(batch_size: int, seq_len: int, embedding_size: int, device: torch.device = None) -> torch.Tensor:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return torch.randn(batch_size, seq_len, embedding_size, device = device)

def precompute_freqs_cis_real(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute RoPE cos/sin of shape [end, dim] for easy broadcasting.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, freqs)  # [end, dim//2]

    # Interleave to full dimension
    cos = torch.zeros(end, dim)
    sin = torch.zeros(end, dim)
    cos[:, 0::2] = freqs.cos()
    cos[:, 1::2] = freqs.cos()
    sin[:, 0::2] = freqs.sin()
    sin[:, 1::2] = freqs.sin()

    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input.
    Used for the RoPE 'real' implementation trick.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings using the Sine-Cosine rewrite.
    """
    # Reshape cos/sin for broadcasting: [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out

def step_embed(
    t: int,
    T: int,
    target: torch.Tensor,
    layer: dict,
    layer_type: str,
    input_ids: torch.Tensor,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    requires_update: bool,
    layer_norm: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for embedding layer.
    Returns (mu, mu_word, error).
    """
    word_layer: nn.Embedding = layer["word"]
    
    vocab_size = word_layer.weight.size(0)
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size-1)
         
    mu_word = word_layer(input_ids)
    mu = mu_word 
    mu_norm = layer_norm(mu) if layer_norm is not None else mu

    error = target - mu_norm
        
    if requires_update: 
        with torch.no_grad():
            flat_input_ids = input_ids.reshape(-1)
            flat_update = error.reshape(-1, error.size(-1))
            delta = torch.clamp(local_lr * flat_update, -0.01, 0.01)
            word_layer.weight.data.index_add_(0, flat_input_ids, delta)
            
    if t == T - 1:
           finalize_step(mu, target, error, t, layer_type, energy_fn_name)
    return mu, mu_word, None, error

def step_linear(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    layer: nn.Module,
    lateral_conn: Optional[Any], 
    layer_type: str,
    local_lr: float,
    inference_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module], 
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
            
    if layer_type == "linear_output":
        bu_err = target - F.softmax(mu, dim=-1) 
    else:    
        bu_err = target - mu 

    # project bottom-up error through weights
    error_proj = bu_err @ layer.weight      
    error = error_proj - td_err if td_err is not None else error_proj  
    
    if lateral_conn is not None:
        x = x + inference_lr * lateral_conn.forward(x, error)
        if requires_update:
            lateral_conn.update_weights(x.detach())
    else:
        x = x + inference_lr * error 

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))
    
    # parameter updates for the layer
    if requires_update:
        delta_W = local_lr * torch.einsum("bsv, bsh -> vh", bu_err, x_input.detach())
        delta_W = torch.clamp(delta_W, -0.01, 0.01)
        layer.weight.data.add_(delta_W)
        if layer.bias is not None and update_bias:
            delta_b = local_lr * bu_err.mean(dim=(0, 1))
            layer.bias.data.add_(torch.clamp(delta_b, -0.01, 0.01))

    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type, energy_fn_name)
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
    inference_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    num_heads: int,
    n_embed: int,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
    rope_cache: Tuple[torch.Tensor, torch.Tensor],
    flash: bool = False,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    ):
    """
    Predictive coding step for attention using Sine-Cosine RoPE.
    - proj_layers must contain 'q_proj','k_proj','v_proj' modules
    """
    assert proj_layers is not None, "proj_layers dict is required for attention"

    device = x.device
    x_norm = layer_norm(x) if layer_norm is not None else x

    q_proj = proj_layers["q_proj"]
    k_proj = proj_layers["k_proj"]
    v_proj = proj_layers["v_proj"]
    assert q_proj is not None and k_proj is not None and v_proj is not None, "Missing Q/K/V projections"  
        
    B, S, E = target.shape
    head_dim = n_embed // num_heads

    #RAW projections (USED FOR LEARNING)
    Q_raw = q_proj(x_norm).view(B, num_heads, S, head_dim)
    K_raw = k_proj(x_norm).view(B, num_heads, S, head_dim)
    V_raw = v_proj(x_norm).view(B, num_heads, S, head_dim)

    #ROTATED copies (USED FOR ATTENTION ONLY)
    Q = Q_raw.clone()
    K_new = K_raw.clone()
    V_new = V_raw  

    cos, sin = rope_cache
    cos = cos.to(device)
    sin = sin.to(device)

    Q, K_new = apply_rotary_emb(Q, K_new, cos[:S], sin[:S])

    #KV cache handling
    if use_cache and kv_cache is not None:
        K_cached, V_cached = kv_cache
        K = torch.cat([K_cached, K_new], dim=2)
        V = torch.cat([V_cached, V_new], dim=2)
    else:
        K, V = K_new, V_new

    causal_mask = torch.tril(
        torch.ones(S, K.size(2), device=device)
    ).unsqueeze(0).unsqueeze(0)

    # !! Causal Mask
    if flash:
        mu_heads = apply_flash_attention(Q, K, V, mask=causal_mask)
    else:
        mu_heads = apply_standard_attention(Q, K, V, mask=causal_mask)

    mu = mu_heads.transpose(1, 2).contiguous().view(B, S, E)

    bu_err = target - mu
    error = bu_err - td_err if td_err is not None else bu_err

    # State update
    if lateral_conn is not None:
        x = x + inference_lr * lateral_conn.forward(x, error)
        if requires_update:
            lateral_conn.update_weights(x.detach())
    else:
        x = x + inference_lr * error

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    #PC WEIGHT UPDATE (RAW, PRE-RoPE)
    if requires_update:
        with torch.no_grad():
            for h in range(num_heads):
                qh = Q_raw[:, h]
                kh = K_raw[:, h]
                vh = V_raw[:, h]

                dWq = torch.einsum("bsd,bse->de", qh, x_norm) / (B * S)
                dWk = torch.einsum("bsd,bse->de", kh, x_norm) / (B * S)
                dWv = torch.einsum("bsd,bse->de", vh, x_norm) / (B * S)

                s, e = h * head_dim, (h + 1) * head_dim

                q_proj.weight.data[s:e] += torch.clamp(local_lr * dWq, -clamp_value, clamp_value)
                k_proj.weight.data[s:e] += torch.clamp(local_lr * dWk, -clamp_value, clamp_value)
                v_proj.weight.data[s:e] += torch.clamp(local_lr * dWv, -clamp_value, clamp_value)

    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type, energy_fn_name)

    return x, mu, bu_err, (K.detach(), V.detach()) if use_cache else None

ENERGY_FUNCTIONS = {
    "pc_e": lambda mu, x: ((mu - x) ** 2) * 0.5,    
    "kld": lambda mu, x: torch.clamp(
        F.kl_div(mu.log_softmax(dim=-1), x, reduction="batchmean"), min=0.0, max=100.0
    ),
}

def energy_fn(mu: torch.Tensor, x: torch.Tensor, energy_fn_name: str) -> torch.Tensor:
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
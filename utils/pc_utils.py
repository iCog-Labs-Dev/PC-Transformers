import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from typing import Optional, Tuple, Any
from utils.attention_utils import apply_flash_attention, apply_standard_attention
from utils.optim.optim_utils import PCOptimizer
    
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

def rotate_half_transpose(x: torch.Tensor) -> torch.Tensor:
    """
    Transpose of rotate_half operation.
    Forward: [x1, x2] -> [-x2, x1]
    Transpose: [y1, y2] -> [y2, -y1]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)

def step_embed(
    t: int,
    T: int,
    target: torch.Tensor,
    layer: dict,
    layer_type: str,
    input_ids: torch.Tensor,
    local_lr: float,
    clamp_value: float,
    clip_value: float,
    energy_fn_name: str,
    requires_update: bool,
    layer_norm: Optional[nn.Module] = None,
    optimizer: Optional[PCOptimizer] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for embedding layer.
    Returns (mu, mu_word, error).
    """
    word_layer: nn.Embedding = layer["word"]
         
    mu_word = word_layer(input_ids)
    mu = mu_word 
    
    error = target - mu
        
    if requires_update: 
        with torch.no_grad():
            flat_input_ids = input_ids.reshape(-1)
            flat_update = error.reshape(-1, error.size(-1))
            if optimizer is not None:
                update_word = torch.zeros_like(word_layer.weight)
                update_word.index_add_(0, flat_input_ids, flat_update)
                optimizer.step_param(word_layer.weight, update_word, local_lr, clip_value=clip_value)
            else:
                delta = torch.clamp(local_lr * flat_update, -0.01, 0.01)
                word_layer.weight.data.index_add_(0, flat_input_ids, delta)

    return mu, mu_word, error
    
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
    clip_value: float,
    energy_fn_name: str,
    requires_update: bool,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module], 
    optimizer: Optional[PCOptimizer] = None,
   ):
    """
    Predictive coding step for linear-like layers.
    Returns: (updated_x, mu, bu_err)
    """
    # if layer_norm is not None and layer_type == "fc1":
    #     x_input = layer_norm(x)
    if layer_type == "fc2":
        x_input = F.gelu(x)
    else:
        x_input = x
        
    mu = layer(x_input)
            
    if layer_type=="linear_output":
        probs  = F.softmax(mu, dim=-1)
        bu_err = target - probs   # reconstruction error (probability space)
        dE_dmu = bu_err   # CE gradient w.r.t. logits

    else:    
        bu_err = target - mu 
        dE_dmu = bu_err
        
    error_proj= dE_dmu @ layer.weight       
    error = error_proj- td_err if td_err is not None else error_proj  
   
    if lateral_conn is not None:
        x = x + inference_lr * lateral_conn.forward(x, error)
        if requires_update:
            lateral_conn.update_weights(x.detach(), optimizer=optimizer, clip_value=clip_value)
    else:
        x = x + inference_lr * error 

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))
    
    # parameter updates for the layer
    if requires_update:
        B, S, _ = dE_dmu.shape
        update_w = torch.einsum("bsv, bsh -> vh", dE_dmu, x_input.detach()) / (B * S)
        if optimizer is not None:
            optimizer.step_param(layer.weight, update_w, local_lr, clip_value=clip_value)
        else:
            delta_W = torch.clamp(local_lr * update_w, -0.01, 0.01)
            layer.weight.data.add_(delta_W)
        if layer.bias is not None:
            update_b = dE_dmu.mean(dim=(0, 1))
            if optimizer is not None:
                optimizer.step_param(layer.bias, update_b, local_lr, clip_value=clip_value)
            else:
                delta_b = torch.clamp(local_lr * update_b, -0.01, 0.01)
                layer.bias.data.add_(delta_b)

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
    clip_value: float,
    energy_fn_name: str,
    requires_update: bool,
    num_heads: int,
    n_embed: int,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
    rope_cache: Tuple[torch.Tensor, torch.Tensor],
    flash: bool = False,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    optimizer: Optional[PCOptimizer] = None,
    ):
    """
    Predictive coding step for attention using Sine-Cosine RoPE.
    - proj_layers must contain 'q_proj','k_proj','v_proj' modules
    """
    assert proj_layers is not None, "proj_layers dict is required for attention"

    device = x.device
    x_norm = x #if layer_norm is not None else x

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
        mu_heads, attn_weights, attn_scores = apply_standard_attention(Q, K, V, mask=causal_mask)
    
    mu = mu_heads.transpose(1, 2).contiguous().view(B, S, E)
    bu_err = target - mu
    
    # deleted to insert the delta_x after the for loop below
    # error = bu_err - td_err if td_err is not None else bu_err  
     
    scale = 1.0 / (head_dim ** 0.5)

    # Backward pass (manual gradients)
    dE_dmu = bu_err  
    dE_dmu_heads = dE_dmu.view(B, num_heads, S, head_dim)
    
    # dE/dV
    dE_dV = torch.matmul(attn_weights.transpose(-2, -1), dE_dmu_heads)

    # dE/dA = dE/dμ @ V^T
    dE_dA = torch.matmul(dE_dmu_heads, V.transpose(-2, -1))

    # Softmax vector-Jacobian product
    norm_term = (dE_dA * attn_weights).sum(dim=-1, keepdim=True)
    dE_dS = attn_weights * (dE_dA - norm_term)
    
    # Apply causal mask to gradients
    if causal_mask is not None:
        dE_dS = dE_dS.masked_fill(causal_mask == 0, 0.0)

    # Gradients through Q and K
    dE_dQ = torch.matmul(dE_dS, K) * scale
    dE_dK = torch.matmul(dE_dS.transpose(-2, -1), Q) * scale

    # Gradients through RoPE (using transpose)
    cos_q = cos[:S].unsqueeze(0).unsqueeze(0)
    sin_q = sin[:S].unsqueeze(0).unsqueeze(0)
    
    K_len = K.size(2)
    cos_k = cos[:K_len].unsqueeze(0).unsqueeze(0)
    sin_k = sin[:K_len].unsqueeze(0).unsqueeze(0)
    
    dE_dQ_raw = (dE_dQ * cos_q) + (rotate_half_transpose(dE_dQ) * sin_q)
    dE_dK_raw = (dE_dK * cos_k) + (rotate_half_transpose(dE_dK) * sin_k)
    
    delta_x = torch.zeros_like(x_norm)

    # Update x per head
    for h in range(num_heads):
        dq = dE_dQ_raw[:, h]        # [B, S, head_dim]
        dk = dE_dK_raw[:, h]        # [B, K_len, head_dim]
        dv = dE_dV[:, h]            # [B, K_len, head_dim]
        
        if dk.size(1) > S:
            dk = dk[:, -S:, :]
            dv = dv[:, -S:, :]
        
        # Correctly slice along dim 0 (out_features). Shape becomes [head_dim, E]
        wq = q_proj.weight[h*head_dim:(h+1)*head_dim, :]
        wk = k_proj.weight[h*head_dim:(h+1)*head_dim, :]
        wv = v_proj.weight[h*head_dim:(h+1)*head_dim, :]
        
        # Adjust einsum to match the new shape: 'he' represents [head_dim, E]
        delta_q = torch.einsum('bsh,he->bse', dq, wq)
        delta_k = torch.einsum('bsh,he->bse', dk, wk)
        delta_v = torch.einsum('bsh,he->bse', dv, wv)
        
        delta_x += delta_q + delta_k + delta_v

    # Update delta_x with TD error if provided
    delta_x = delta_x - td_err if td_err is not None else delta_x

    if lateral_conn is not None:
        delta_x = lateral_conn.forward(x, delta_x)
        x = x + inference_lr * delta_x
        
        if requires_update:
             lateral_conn.update_weights(x.detach(), optimizer=optimizer, clip_value=clip_value)
    else:
        x = x + inference_lr * delta_x

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    if requires_update:
        with torch.no_grad():
            update_q = torch.zeros_like(q_proj.weight)
            update_k = torch.zeros_like(k_proj.weight)
            update_v = torch.zeros_like(v_proj.weight)

            update_b_q = torch.zeros_like(q_proj.bias) if q_proj.bias is not None else None
            update_b_k = torch.zeros_like(k_proj.bias) if k_proj.bias is not None else None
            update_b_v = torch.zeros_like(v_proj.bias) if v_proj.bias is not None else None

            for h in range(num_heads):
                # Swap einsum output to 'de' to get shape [head_dim, E] matching the weight slice
                # Apply updates to dim 0 (out_features)
                update_q[h*head_dim:(h+1)*head_dim, :] = torch.clamp(torch.einsum("btd,bte->de", dE_dQ_raw[:, h], x_norm), -0.01, 0.01)
                update_k[h*head_dim:(h+1)*head_dim, :] = torch.clamp(torch.einsum("btd,bte->de", dE_dK_raw[:, h], x_norm), -0.01, 0.01)
                update_v[h*head_dim:(h+1)*head_dim, :] = torch.clamp(torch.einsum("btd,bte->de", dE_dV[:, h], x_norm), -0.01, 0.01)

                if update_b_q is not None:
                    update_b_q[h*head_dim:(h+1)*head_dim] = torch.clamp(dE_dQ_raw[:, h].mean(dim=(0, 1)), -0.01, 0.01)
                if update_b_k is not None:
                    update_b_k[h*head_dim:(h+1)*head_dim] = torch.clamp(dE_dK_raw[:, h].mean(dim=(0, 1)), -0.01, 0.01)
                if update_b_v is not None:
                    update_b_v[h*head_dim:(h+1)*head_dim] = torch.clamp(dE_dV[:, h].mean(dim=(0, 1)), -0.01, 0.01)

            if optimizer is not None:
                optimizer.step_param(q_proj.weight, update_q, local_lr, clip_value=clip_value)
                optimizer.step_param(k_proj.weight, update_k, local_lr, clip_value=clip_value)
                optimizer.step_param(v_proj.weight, update_v, local_lr, clip_value=clip_value)
                if update_b_q is not None:
                    optimizer.step_param(q_proj.bias, update_b_q, local_lr, clip_value=clip_value)
                if update_b_k is not None:
                    optimizer.step_param(k_proj.bias, update_b_k, local_lr, clip_value=clip_value)
                if update_b_v is not None:
                    optimizer.step_param(v_proj.bias, update_b_v, local_lr, clip_value=clip_value)
            else:
                q_proj.weight.data.add_(torch.clamp(local_lr * update_q, -0.01, 0.01))
                k_proj.weight.data.add_(torch.clamp(local_lr * update_k, -0.01, 0.01))
                v_proj.weight.data.add_(torch.clamp(local_lr * update_v, -0.01, 0.01))
                if update_b_q is not None:
                    q_proj.bias.data.add_(torch.clamp(local_lr * update_b_q, -0.01, 0.01))
                if update_b_k is not None:
                    k_proj.bias.data.add_(torch.clamp(local_lr * update_b_k, -0.01, 0.01))
                if update_b_v is not None:
                    v_proj.bias.data.add_(torch.clamp(local_lr * update_b_v, -0.01, 0.01))
    new_kv_cache = (K.detach(), V.detach()) if use_cache else None
    return x, mu, bu_err, new_kv_cache

ENERGY_FUNCTIONS = {
    "pc_e": lambda mu, x: ((x - mu) ** 2) * 0.5,
    # Added: CE energy for output layer
    "ce": lambda mu, x: F.cross_entropy(
        mu.reshape(-1, mu.size(-1)),
        x.argmax(dim=-1).reshape(-1),
        reduction="mean",
    ),   
    "kld": lambda mu, x: torch.clamp(
        F.kl_div(mu.log_softmax(dim=-1), x, reduction="batchmean"), min=0.0, max=100.0
    ),
}

def energy_fn(mu: torch.Tensor, x: torch.Tensor,energy_fn_name: str) -> torch.Tensor:
    if energy_fn_name not in ENERGY_FUNCTIONS:
        raise ValueError(f"Unknown energy function: {energy_fn_name}. Choose from {list(ENERGY_FUNCTIONS.keys())}")
    return ENERGY_FUNCTIONS[energy_fn_name](mu, x)

def finalize_step(mu: torch.Tensor, target: torch.Tensor, error: torch.Tensor, t: int, layer_type: str, energy_fn_name: str, output_energy_fn_name: str = "ce"): # added: CE for output layer
    device = mu.device
    target = target.to(device)
    error = error.to(device)
    # Route output layer to CE, hidden layers to pc_e
    fn_name = (
        output_energy_fn_name     # "ce"   — linear_output
        if layer_type == "linear_output"
        else energy_fn_name       # "pc_e" — all hidden layers
    )
    energy = float(energy_fn(mu, target, fn_name).sum().item())
    errors = [{"step": t, "type": layer_type, "error": error.mean().item()}]
    return energy, errors
    
def ids_to_one_hot(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    device = input_ids.device
    return F.one_hot(input_ids, num_classes=vocab_size).float().to(device)

def cleanup_memory():
    """Comprehensive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
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

def init_x_query(batch_size: int, seq_len: int, n_embed: int, device: torch.device = None) -> torch.Tensor:
    return x_init(batch_size, seq_len, n_embed, device)

def init_x_key(batch_size: int, seq_len: int, n_embed: int, device: torch.device = None) -> torch.Tensor:
    return x_init(batch_size, seq_len, n_embed, device)

def init_x_value(batch_size: int, seq_len: int, n_embed: int, device: torch.device = None) -> torch.Tensor:
    return x_init(batch_size, seq_len, n_embed, device)

def init_x_score(batch_size: int, seq_len: int, n_embed: int, device: torch.device = None) -> torch.Tensor:
    return x_init(batch_size, seq_len, n_embed, device)

def init_x_A(batch_size: int, seq_len: int, n_embed: int, device: torch.device = None) -> torch.Tensor:
    return x_init(batch_size, seq_len, n_embed, device)

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
        update_w = torch.einsum("bsv, bsh -> vh", bu_err, x_input.detach())
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

def _step_projected_latent(
    *,
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    layer: nn.Linear,
    lateral_conn: Any,
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
    optimizer: Optional[PCOptimizer],
):
    x_input = layer_norm(x) if layer_norm is not None else x
    mu = layer(x_input)
    bu_err = target - mu

    error_proj = bu_err @ layer.weight
    error = error_proj - td_err if td_err is not None else error_proj

    # Always use lateral connections when updating
    if lateral_conn is None:
        raise ValueError(f"Lateral connection is required for layer_type={layer_type}")

    delta_x = lateral_conn.forward(x, error)
    x = x + local_lr * delta_x
    if requires_update:
        lateral_conn.update_weights(x.detach(), optimizer=optimizer, clamp_value=0.01)

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    if requires_update:
        update_w = torch.einsum("bsv, bsh -> vh", bu_err, x_input.detach())
        if optimizer is not None:
            optimizer.step_param(layer.weight, update_w, local_lr, clamp_value=0.01)
        else:
            layer.weight.data.add_(torch.clamp(local_lr * update_w, -0.01, 0.01))

        if layer.bias is not None and update_bias:
            update_b = bu_err.mean(dim=(0, 1))
            if optimizer is not None:
                optimizer.step_param(layer.bias, update_b, local_lr, clamp_value=0.01)
            else:
                layer.bias.data.add_(torch.clamp(local_lr * update_b, -0.01, 0.01))

    if t == T - 1:
        finalize_step(mu, target, bu_err, t, layer_type, energy_fn_name)

    return x, mu, bu_err


def step_x_query(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    layer: nn.Linear,
    lateral_conn: Any,
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
    return _step_projected_latent(
        t=t,
        T=T,
        target=target,
        x=x,
        layer=layer,
        lateral_conn=lateral_conn,
        layer_type=layer_type,
        local_lr=local_lr,
        clamp_value=clamp_value,
        energy_fn_name=energy_fn_name,
        update_bias=update_bias,
        requires_update=requires_update,
        td_err=td_err,
        layer_norm=layer_norm,
        optimizer=optimizer,
    )


def step_x_key(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    layer: nn.Linear,
    lateral_conn: Any,
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
    return _step_projected_latent(
        t=t,
        T=T,
        target=target,
        x=x,
        layer=layer,
        lateral_conn=lateral_conn,
        layer_type=layer_type,
        local_lr=local_lr,
        clamp_value=clamp_value,
        energy_fn_name=energy_fn_name,
        update_bias=update_bias,
        requires_update=requires_update,
        td_err=td_err,
        layer_norm=layer_norm,
        optimizer=optimizer,
    )


def step_x_value(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    layer: nn.Linear,
    lateral_conn: Any,
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
    return _step_projected_latent(
        t=t,
        T=T,
        target=target,
        x=x,
        layer=layer,
        lateral_conn=lateral_conn,
        layer_type=layer_type,
        local_lr=local_lr,
        clamp_value=clamp_value,
        energy_fn_name=energy_fn_name,
        update_bias=update_bias,
        requires_update=requires_update,
        td_err=td_err,
        layer_norm=layer_norm,
        optimizer=optimizer,
    )


def step_x_score(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    layer: nn.Linear,
    lateral_conn: Any,
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module] = None,
    optimizer: Optional[PCOptimizer] = None,
):
    x_input = layer_norm(x) if layer_norm is not None else x
    mu = layer(x_input)
    bu_err = target - mu

    error_proj = bu_err @ layer.weight
    error = error_proj - td_err if td_err is not None else error_proj

    if lateral_conn is None:
        raise ValueError(f"Lateral connection is required for layer_type={layer_type}")

    delta_x = lateral_conn.forward(x, error)
    x = x + local_lr * delta_x
    if requires_update:
        lateral_conn.update_weights(x.detach(), optimizer=optimizer, clamp_value=0.01)

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    if requires_update:
        update_w = torch.einsum("bsv, bsh -> vh", bu_err, x_input.detach())
        if optimizer is not None:
            optimizer.step_param(layer.weight, update_w, local_lr, clamp_value=0.01)
        else:
            layer.weight.data.add_(torch.clamp(local_lr * update_w, -0.01, 0.01))

        if layer.bias is not None and update_bias:
            update_b = bu_err.mean(dim=(0, 1))
            if optimizer is not None:
                optimizer.step_param(layer.bias, update_b, local_lr, clamp_value=0.01)
            else:
                layer.bias.data.add_(torch.clamp(local_lr * update_b, -0.01, 0.01))

    if t == T - 1:
        finalize_step(mu, target, bu_err, t, layer_type, energy_fn_name)
    return x, mu, bu_err


def step_x_A(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    layer: nn.Linear,
    lateral_conn: Any,
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    x_score: torch.Tensor,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module] = None,
    optimizer: Optional[PCOptimizer] = None,
):
    # step_x_A: x is initialized/overwritten as softmax(x_score) (as requested)
    x = torch.softmax(x_score, dim=-1)
    x_input = layer_norm(x) if layer_norm is not None else x
    mu = layer(x_input)
    bu_err = target - mu

    error_proj = bu_err @ layer.weight
    error = error_proj - td_err if td_err is not None else error_proj

    if lateral_conn is None:
        raise ValueError(f"Lateral connection is required for layer_type={layer_type}")

    delta_x = lateral_conn.forward(x, error)
    x = x + local_lr * delta_x
    if requires_update:
        lateral_conn.update_weights(x.detach(), optimizer=optimizer, clamp_value=0.01)

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    if requires_update:
        update_w = torch.einsum("bsv, bsh -> vh", bu_err, x_input.detach())
        if optimizer is not None:
            optimizer.step_param(layer.weight, update_w, local_lr, clamp_value=0.01)
        else:
            layer.weight.data.add_(torch.clamp(local_lr * update_w, -0.01, 0.01))

        if layer.bias is not None and update_bias:
            update_b = bu_err.mean(dim=(0, 1))
            if optimizer is not None:
                optimizer.step_param(layer.bias, update_b, local_lr, clamp_value=0.01)
            else:
                layer.bias.data.add_(torch.clamp(local_lr * update_b, -0.01, 0.01))

    if t == T - 1:
        finalize_step(mu, target, bu_err, t, layer_type, energy_fn_name)
    return x, mu, bu_err
    
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
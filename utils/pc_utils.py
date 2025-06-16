import torch
import torch.nn.functional as F
import math
from predictive_coding.config import GPTConfig

def compute_DVL(attn_v):
    B, H, T, D= attn_v.shape
    
    #print(f"Head Outputs Variance: {attn_v.var().item():.4f}")
    # Ensure input tensor has requires_grad=True
    x= attn_v.transpose(0, 1).flatten(2, 3)
    x=F.normalize(x, p=2, dim=-1)
    s_m=torch.bmm(x, x.transpose(1, 2))
    N = s_m.size(1)
    mask = ~torch.eye(N, device=x.device).bool()
    s_m= s_m[:, mask].view(H, H, -1)
    s_m= s_m.mean(dim=-1)
    identity = torch.eye(s_m.size(1), device=s_m.device)
    identity = identity.unsqueeze(0).expand(H, -1, -1)
    
    corr=  s_m - identity
    dvl=(corr** 2).mean()
    
   
    try:
        dvl_grad= torch.autograd.grad(dvl, 
                                      attn_v,
                                      retain_graph= True,
                                      )[0]
    except Exception as e:
        print(f" Error computing diversity gradient: {e}")
        dvl_grad=torch.zeros_like(attn_v)
        
    
    return dvl_grad


def get_head_similarity(mu_heads):
    B, H, T, D = mu_heads.shape
    x = mu_heads.transpose(0, 1).flatten(2, 3)  # [H, N, D]
    x = F.normalize(x, p=2, dim=-1)
    
    # Compute pairwise cosine similarity between heads
    corr = torch.bmm(x, x.transpose(1, 2))  # [H, N, N]
    mask = ~torch.eye(corr.size(1), device=corr.device).bool()
    s_v = corr[:, mask].view(H, H, -1)
    corr = s_v.abs().mean(dim=-1)  # [H, H]

    return corr.detach().cpu()
    
def x_init(batch_size: int, seq_len: int, embedding_size: int) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, embedding_size)

def step_embed(t, T, target, layer, layer_type, input_ids, position_ids, local_lr, clamp_value, energy_fn_name, is_holding_error, requires_update):
    word_layer = layer["word"]
    pos_layer = layer["pos"]

    mu_word = word_layer(input_ids)
    mu_pos = pos_layer(position_ids)
    mu = mu_word + mu_pos
    error = target - mu

    update = torch.clamp(error, -clamp_value, clamp_value)
    if requires_update:
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
    
def step_linear(t, T, target, x, layer, W_latents, layer_type, local_lr, clamp_value, use_lateral, is_holding_error, energy_fn_name, update_bias, requires_update):
        mu = layer(x)
        if layer_type == "fc1":
            mu = F.gelu(mu)

        error = target - mu
        if layer.weight.shape[0] != layer.weight.shape[1]:
            error_proj = torch.einsum("bsh, vh -> bsv", error, layer.weight.T)  
        else:
            error_proj = error  

        if use_lateral and layer_type in W_latents:
            W_latent = W_latents[layer_type]
            x_latent = torch.einsum("bsh,hv->bsv", x, W_latent)
            delta_x = error_proj + x_latent
            x = x + local_lr * delta_x

            if requires_update:
                anti_hebbian_latent = -torch.einsum("bsh,bsv->hv", x.detach(), x.detach())
                W_latents[layer_type].data.add_(local_lr * anti_hebbian_latent)
        
        else:
            x= x + local_lr * error 
        
        x = torch.clamp(x, -clamp_value, clamp_value)
        
        # Hebbian Update W_layer
        if requires_update:
            delta_W = local_lr * torch.einsum("bsh,bsv->hv", error, x.detach())
            layer.weight.data.add_(delta_W)

            if layer.bias is not None and update_bias:
                layer.bias.data.add_(local_lr * error.mean(dim=(0, 1)))

        if t == T - 1:
            finalize_step(mu, target, error, t, layer_type,energy_fn_name, is_holding_error)

        return x, mu

def step_attn(t, T, target, x, W_latents, proj_layers, layer_type, local_lr, clamp_value, use_lateral, is_holding_error, energy_fn_name, update_bias, requires_update, layer_instance):
        assert proj_layers is not None, "proj_layers dict is required for attention"
        q_proj = proj_layers.get("q_proj", None)
        k_proj = proj_layers.get("k_proj", None)
        v_proj = proj_layers.get("v_proj", None)
        
        assert all(p is not None for p in (q_proj, k_proj, v_proj)), "Missing Q/K/V projections in dict"        
        Q= q_proj(x)
        K= k_proj(x)
        V= v_proj(x)
        batch_size, seq_len, embed_dim=target.shape
        
        num_heads = GPTConfig.num_heads
        head_dim = GPTConfig.n_embed // GPTConfig.num_heads 
        la= GPTConfig.la * math.sqrt(1.0 / head_dim)

        Q = Q.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2) # B. H, T, D
        K = K.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        V = V.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

          
        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1)) #B,H,T,T
        mask = torch.tril(torch.ones_like(scores, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = scores.softmax(dim=-1) # B, H, T, T
        mu_heads = attn_weights @ V   # B, H, T, D
        dvl_grad=compute_DVL(mu_heads)
        #dvl_norm = dvl_grad.norm().item()
    
        #print(f"Diversity Grad Norm: {dvl_norm:.8f}")
        similarity = get_head_similarity(mu_heads)
        #print(f"Headwise Similarity Matrix:\n{similarity.numpy()}")
        
        mu = mu_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
     
        error = target - mu  # B, T, D
        #print("error shape:", error.shape)
        #print("dvl_grad shape:", dvl_grad.shape)

        if dvl_grad is not None:
            B, T, H, D = dvl_grad.shape
            dvl_projected = dvl_grad.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
            #print("dvl_grad shape:", dvl_projected.shape)
            dvl_projected=dvl_projected.clamp(-1e-3, 1e-3)
            error = error + la * dvl_projected
        else:
            error = error
        if layer_instance is not None:
            setattr(layer_instance, '_head_similarity', similarity)
            setattr(layer_instance, '_head_similarity_avg', similarity.mean().item())
            setattr(layer_instance, '_head_similarity_max', similarity.max().item())
    

        if use_lateral and layer_type in W_latents:
            W_latent = W_latents[layer_type]
            x_latent = x @ W_latent
            delta_x = error + x_latent
            x = x + local_lr * delta_x

            if requires_update:
                anti_hebbian_latent = - torch.einsum("bsh,bsv->hv", x.detach(), x.detach())
                W_latents[layer_type].data.add_(local_lr * anti_hebbian_latent)
            
        else:
            x= x+ local_lr * error

        x = torch.clamp(x, -clamp_value, clamp_value)

        # Hebbian update W_latent
        if requires_update:
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
    "cosine": lambda mu, x: 1 - F.cosine_similarity(mu, x, dim=-1),
    "kld": lambda mu, x: F.kl_div(
        mu.log_softmax(dim=-1),
        x.softmax(dim=-1),
        reduction='batchmean'
    )
}

def energy_fn(mu: torch.Tensor, x: torch.Tensor,energy_fn_name: str) -> torch.Tensor:
    if energy_fn_name not in ENERGY_FUNCTIONS:
        raise ValueError(f"Unknown energy function: {energy_fn_name}. Choose from {list(ENERGY_FUNCTIONS.keys())}")
    return ENERGY_FUNCTIONS[energy_fn_name](mu, x)

def finalize_step(mu, target, error, t, layer_type,energy_fn_name, is_holding_error = False):
    energy = energy_fn(mu, target,energy_fn_name).mean().item() if is_holding_error else None
    errors = [{"step": t, "type": layer_type, "error": error.mean().item()}]
    return energy, errors
    
def ids_to_one_hot(input_ids, vocab_size):
        """input_id from [B, S] to [B, S, V]"""
        return F.one_hot(input_ids, num_classes=vocab_size).float()
import torch
from torch.amp import autocast
import logging

# Set up logging
logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    level=logging.INFO 
)

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    FLASH_AVAILABLE = True
    logging.info("FlashAttention is available and will be used.")
except ImportError:
    FLASH_AVAILABLE = False
    logging.warning("FlashAttention is not installed. Falling back to standard attention.")
    
device = "cuda" if torch.cuda.is_available() else "cpu"

def apply_flash_attention(q, k, v, mask=None):
    """
    Apply FlashAttention if available, else fallback to standard attention.
    Args:
        q, k, v: Query, Key, Value tensors (B, num_heads, T, head_dim)
        mask: Optional mask tensor
    Returns:
        attn_output: Output tensor after attention
    """
    if not FLASH_AVAILABLE:
        return apply_standard_attention(q, k, v, mask)
    B, num_heads, T, head_dim = q.shape
    # FlashAttention expects [B, T, 3, num_heads, head_dim]
    qkv = torch.stack([q, k, v], dim=2) # [B, T, 3, num_heads, head_dim]
    orig_dtype = qkv.dtype
    with autocast(device_type=device, dtype=torch.float16):
        if qkv.dtype not in [torch.float16, torch.bfloat16]:
            qkv = qkv.to(torch.float16)
        attn_out = flash_attn_qkvpacked_func(qkv, 0.0, None, causal=True)
        attn_out = attn_out.to(orig_dtype)
    # Output: [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
    return attn_out

def apply_standard_attention(q, k, v, mask=None):
    """
    Standard scaled dot-product attention with masking and mixed precision.
    Args:
        q, k, v: Query, Key, Value tensors (B, num_heads, T, head_dim)
        mask: Optional mask tensor
    Returns:
        attn_output: Output tensor after attention
    """
    with autocast(device_type=device, dtype=torch.float16):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
    return attn_output

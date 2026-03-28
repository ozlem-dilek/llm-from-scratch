import torch
import torch.nn as nn


def _rope_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    """θ_i = base^(-2i/d), i = 0, ..., d/2 - 1"""
    half = head_dim // 2
    idx = torch.arange(0, half, dtype=torch.float32)
    return base ** (-2.0 * idx / float(head_dim))


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (B, H, T, D) — D çift
    cos, sin: (1, 1, T, D//2)
    (x_{2i}, x_{2i+1}) -> (x_{2i}*c_i - x_{2i+1}*s_i, x_{2i}*s_i + x_{2i+1}*c_i)
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = y1
    out[..., 1::2] = y2
    return out


class RoPE(nn.Module):
    #rotary position embedding — inv_freq buffer; cos/sin forward'da üretilir.

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE için head_dim çift olmalı")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv = _rope_inv_freq(head_dim, base)
        self.register_buffer("inv_freq", inv)

    def forward(self, seq_len: int, device, dtype) -> tuple:
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len ({seq_len}) > max_seq_len ({self.max_seq_len})")
        t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        inv = self.inv_freq.to(device=device, dtype=torch.float32)
        angles = t * inv
        cos = angles.cos().to(dtype=dtype)
        sin = angles.sin().to(dtype=dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return cos, sin

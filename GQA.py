import torch
import torch.nn as nn
import torch.nn.functional as F

from RoPE import apply_rope


class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        #Q için tüm headler, K ve V için sadece n_kv_heads kadar

        self.wq = nn.Linear(d_model, n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads*self.head_dim, d_model, bias=False)

        #mask
        self.register_buffer("mask", torch.tril(torch.ones(2048,2048)).view(1,1,2048,2048))

    def forward(self, x, rope):
        B, T, C = x.size()

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = rope(T, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # k ve v headlerini q'ya eşitlemek için expand
        num_kv_groups = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)

        #pytorch 2.0 flashattention (otomatik oalrak SRAM optimize çalışır)
        #bu tek satır comp. O(N)'e düşürür.

        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B,T,C)

        return self.wo(y)


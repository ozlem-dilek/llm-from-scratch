import torch
import torch.nn as nn
from GQA import GQA
from RMSNorm import RMSNorm
from SwiGLU import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, hidden_dim):
        super().__init__()

        self.attention = GQA(d_model, n_heads, n_kv_heads)
        self.feed_forward = SwiGLU(d_model, hidden_dim)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, rope):  # pre-norm
        x = x + self.attention(self.norm1(x), rope)
        x = x + self.feed_forward(self.norm2(x))

        return x
    

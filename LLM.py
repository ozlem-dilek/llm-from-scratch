from RMSNorm import RMSNorm
from TransformerBlock import TransformerBlock
from RoPE import RoPE
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_kv_heads, hidden_dim, max_seq_len):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)
        head_dim = d_model // n_heads
        self.rope = RoPE(head_dim, max_seq_len)
        self.drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, hidden_dim) for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std = 0.02)
        
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('w_down.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*self.n_layers))
    
    def forward(self, idx, targets=None):
        B, T = idx.size()

        x = self.token_emb(idx)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, self.rope)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss
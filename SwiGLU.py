import torch
import torch.nn as nn
import torch.nn.functional as F

#klasik 2 katmanlı mlp yerine 3 matrisli gating mlp

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()

        #gelen veriyi 2 farklı yola ayırıyoruz.

        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x): #swish (silu) actv. ile gate hesapla, up ile çarp, down ile sıkıştır.
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)

        return self.w_down(gate*up)

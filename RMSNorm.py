import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x): #x'in karelerinin ortalamasını al, kök içine koy ve x'i buna böl
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

        return self.weight* norm_x

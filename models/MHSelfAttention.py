from pickle import NONE
import numpy as np
import torch
from einops import rearrange
from torch import nn

class MHSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, causal=True): 
        super().__init__()
        self.dim_head = (int(dim/heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.causal = causal
        self.to_qkv = nn.Linear(dim, _dim * 3, bias=False)
        self.W_out = nn.Linear( _dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5
        self.causal = causal

    def set_causal(self, causal):
        self.causal = causal

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qkv = self.to_qkv(x) 

        q, k, v = tuple(rearrange(qkv, 'b n (d k h) -> k b h n d ', k=3, h=self.heads)) 

        scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', q, k) * self.scale_factor
 
        i = scaled_dot_prod.shape[2]
        j = scaled_dot_prod.shape[3]
        if self.causal:
            mask = torch.ones(i, j, device = 'cuda').triu_(j - i+1).bool()

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[2:] 
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1) 

        out = torch.einsum('b h i j , b h j d -> b h i d', attention, v)

        out = rearrange(out, "b h n d -> b n (h d)") 

        return self.W_out(out)
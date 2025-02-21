from math import gcd, ceil
import functools

import torch
from torch import nn, einsum
import torch.nn.functional as F

from models.Rotary_Embedding_torch import RotaryEmbedding, apply_rotary_emb

from einops import rearrange, repeat
import numpy as np


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class PerceiverARAttention(nn.Module):
    def __init__(
        self,
        *,
        dim = 512, 
        heads = 8,
        causal = True,
        sequence_len = 1024,
        latent_len = 256,  
        pos_emb = None,
        dropout = 0.,
        layer_num = 0
    ):
        super().__init__()
        self.sequence_len = sequence_len
        self.latent_len = latent_len
        self.context_len = self.sequence_len - self.latent_len
        assert (self.context_len + latent_len == sequence_len), 'context_length plus latent should be equal to sequence length'
        self.layer_num = layer_num 
        self.dim_head = dim//heads  
        self.scale = self.dim_head ** -0.5  

        self.heads = heads
        self.causal = causal

        self.norm = nn.LayerNorm(self.dim_head) 

        self.pos_emb = default(pos_emb, RotaryEmbedding(self.dim_head))
        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, dim, bias = False) 
        self.to_kv = nn.Linear(dim, dim, bias = False)   
        self.to_out = nn.Linear(dim, dim) 

       


    def forward(self, x, mask = None): 
        b, n, *_, h, device, causal = *x.shape, self.heads, x.device, self.causal, 

        if self.layer_num ==0:
            x_cxt = x[:,0:self.context_len,:]
            x_lat = x[:,self.context_len:,:]
           

        mask_value = -torch.finfo(x.dtype).max

        qkv = (self.to_q(x), self.to_kv(x))  
        padded_len = x.shape[-2]    

        seq_range = torch.arange(padded_len, device = device)

        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)   

        if exists(self.pos_emb):
            rotary_emb = self.pos_emb(seq_range, cache_key = padded_len) 
            rotary_emb = rearrange(rotary_emb, 'n d -> () n d') 
            q, kv = map(lambda t: apply_rotary_emb(rotary_emb, t), (q, kv))

        q_lat = q[:,self.context_len:,:]
        q_ctx = q[:,0:self.context_len,:]
        q = q_lat
        kv_ctx = kv[:,0:self.context_len,:]

        q = q * self.scale 
        q_ctx = q_ctx * self.scale
        lkv = self.norm(kv)
        lkv_ctx = self.norm(kv_ctx)
        lsim = einsum('b i d, b j d -> b i j', q, lkv)  
        lsim_ctx = einsum('b i d, b j d -> b i j', q_ctx, lkv_ctx)  

        m_size = lsim.shape[-2] 

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n') 
            lsim.masked_fill_(~mask, mask_value)

        if self.causal:
            causal_mask = torch.ones(m_size, m_size, device = device).triu_(1).bool()
            lsim[:,:,self.context_len:].masked_fill_(causal_mask, mask_value)
 
        attn = lsim.softmax(dim = -1)
        attn_ctx = lsim_ctx.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        attn_ctx = self.attn_dropout(attn_ctx)

        out = einsum('b i j, b j d -> b i d', attn, lkv)
        out_ctx = einsum('b i j, b j d -> b i d', attn_ctx, lkv_ctx)
        out = rearrange(out, '(b h) n d -> b (n) (h d)', h = h)
        out_ctx = rearrange(out_ctx, '(b h) n d -> b (n) (h d)', h = h)
        out = torch.cat([out,out_ctx],dim=-2)
        return self.to_out(out)


class PerceiverARTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_unique_tokens,  
        dim,   
        latent_len,
        num_layers, 
        heads = 8,
        sequence_len,
        causal = True,
        ff_mult = 4,  
        ff_dropout = 0.,
        attn_dropout = 0.
    ):
        super().__init__()
        self.sequence_len = sequence_len
        self.latent_len = latent_len
     
        self.token_emb = nn.Linear(num_unique_tokens+3, dim)
        self.sig = nn.Tanh()
        self.dim_head = dim//heads
        pos_emb = RotaryEmbedding(self.dim_head)
 
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, PerceiverARAttention(dim = dim, heads = heads, sequence_len = sequence_len, latent_len = latent_len,causal = causal,layer_num=i, pos_emb = pos_emb, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.to_logits = nn.Sequential(  
            nn.LayerNorm(dim),
            nn.Linear(dim, num_unique_tokens),
        )
        
        self.to_value_head = nn.Sequential(  
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )

    def forward(self, x, output_mask = None):
        x = self.token_emb(x)
        for attn, ff in self.layers:

            x = attn(x, mask = None) + x
            x = ff(x) + x  
        return self.to_logits(x)[:,self.latent_len:,:], self.to_value_head(x)[:,self.latent_len:,:] 

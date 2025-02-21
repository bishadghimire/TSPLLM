from torch import nn
from models.TransformerBlock import TransformerBlock
from models.PositionalEncoding import PositionalEncoding
import torch

class SimpleTransformer(nn.Module):
   def __init__(self, dim, num_unique_tokens=256, num_layers=6, heads=8, dim_head=None, 
       context_len=52, node_gen_len=53, max_seq_len=105, causal=True):
       super().__init__()
       self.max_seq_len = max_seq_len-1
       self.causal=causal
      
       self.token_emb = nn.Linear(num_unique_tokens+3, dim) 

       self.pos_enc = PositionalEncoding(dim, context_len=context_len, node_gen_len = node_gen_len, max_seq_length=max_seq_len)

       self.block_list = [TransformerBlock(dim=dim, heads=heads, dim_head=dim_head,causal=causal) for _ in range(num_layers)]
       self.layers = nn.ModuleList(self.block_list)

       self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_unique_tokens)
        )

   def set_causal(self, causal):
       for b in self.block_list:
           b.set_causal(causal)

   def forward(self, x, mask=None):
       x = self.token_emb(x)
       x = x + self.pos_enc(x)
       for layer in self.layers:
           x = layer(x, mask)
       return self.to_logits(x)
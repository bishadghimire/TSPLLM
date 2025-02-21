import torch
from torch import nn
from torch.autograd import Variable
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=105,

        context_len=52, node_gen_len = 53,dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.node_gen_len = node_gen_len
        pe = torch.zeros(node_gen_len, embedding_dim)
        for pos in range(node_gen_len):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/embedding_dim)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/embedding_dim)))
        pe = pe.unsqueeze(0)        
        self.register_buffer('pe', pe) 
    
    def forward(self, x):
        x = x*math.sqrt(self.embedding_dim)
        seq_length = self.node_gen_len 
        pe = Variable(self.pe, requires_grad=False).to(x.device)

        x[:,-self.node_gen_len:] = x[:,-self.node_gen_len:] + pe

        x = self.dropout(x)
        return x
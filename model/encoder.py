import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import positional_encoding
from .feed_foward import FeedForwardNetwork
from .attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, max_seq_len, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        
        x = self.input_proj(x)
        x += positional_encoding(seq_len, self.d_model)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)
            
        return x

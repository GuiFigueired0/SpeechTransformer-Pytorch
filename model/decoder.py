import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import positional_encoding
from .feed_foward import FeedForwardNetwork
from .attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.dropout3 = nn.Dropout(dropout)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3
    
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, dropout):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        )
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        seq_len = x.size(1)
        
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += positional_encoding(seq_len, self.d_model, x.device)

        for layer in self.layers:
            x = layer(x, enc_output, look_ahead_mask, padding_mask)
        x = self.final_layer(x)
        
        return x

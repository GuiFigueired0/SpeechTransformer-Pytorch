import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .pre_net import Pre_Net

class Transformer(nn.Module):
    def __init__(self, num_layers_enc, num_layers_dec, d_model, num_heads, dff, target_vocab_size, dropout):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(num_layers_enc, d_model, num_heads, dff, dropout)
        self.decoder = Decoder(num_layers_dec, d_model, num_heads, dff, target_vocab_size, dropout)

    def forward(self, inputs, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        inp, tar = inputs
        
        enc_output = self.encoder(x=inp, 
                                  mask=enc_padding_mask )
        dec_output = self.decoder(x=tar, 
                                  enc_output=enc_output,
                                  look_ahead_mask=look_ahead_mask,
                                  padding_mask=dec_padding_mask )
        
        return dec_output

class SpeechTransformer(nn.Module):
    def __init__(self, 
                 target_vocab_size,
                 num_M=2, 
                 n=64, 
                 c=64, 
                 num_layers_enc=12, 
                 num_layers_dec=6, 
                 d_model=256, 
                 num_heads=4, 
                 dff=2048, 
                 dropout=0.1 ):
        super(SpeechTransformer, self).__init__()

        self.pre_net = Pre_Net(num_M, n, c)
        self.transformer = Transformer(num_layers_enc, num_layers_dec, d_model, num_heads, dff, target_vocab_size, dropout)

    def forward(self, inputs, targets, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        out = self.pre_net(inputs)
        out = self.transformer( inputs=(out, targets),
                                enc_padding_mask=enc_padding_mask,
                                look_ahead_mask=look_ahead_mask,
                                dec_padding_mask=dec_padding_mask )
        
        return out
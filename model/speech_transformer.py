import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .pre_net import Pre_Net

class Transformer(nn.Module):
    def __init__(self, 
                 num_layers=4, 
                 d_model=512, 
                 num_heads=8, 
                 dff=2048, 
                 pe_max_len=8000,
                 target_vocab_size=8000, 
                 dropout=0.1 ):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_max_len, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_max_len, dropout)

    def forward(self, inputs, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        inp, tar = inputs
        
        enc_output = self.encoder(inputs=inp, 
                                  enc_padding_mask=enc_padding_mask, 
                                  training=training )
        dec_output, attention_weights = self.decoder(inputs=tar, 
                                                     enc_output=enc_output,
                                                     look_ahead_mask=look_ahead_mask,
                                                     padding_mask=dec_padding_mask, 
                                                     training=training )
        
        return dec_output, attention_weights

class SpeechTransformer(nn.Module):
    def __init__(self, 
                 num_M=2, 
                 n=64, 
                 c=64, 
                 num_layers=4, 
                 d_model=512, 
                 num_heads=8, 
                 dff=2048, 
                 pe_max_len=8000,
                 target_vocab_size=8000, 
                 dropout=0.1 ):
        super(SpeechTransformer, self).__init__()

        self.pre_net = Pre_Net(num_M, n, c)
        self.transformer = Transformer(num_layers, d_model, num_heads, dff, pe_max_len, target_vocab_size, dropout)

    def forward(self, inputs, targets, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        out = self.pre_net(inputs, training=training)
        final_out, attention_weights = self.transformer(inputs=(out, targets), 
                                                        training=training,
                                                        enc_padding_mask=enc_padding_mask,
                                                        look_ahead_mask=look_ahead_mask,
                                                        dec_padding_mask=dec_padding_mask )
        
        return final_out, attention_weights
import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .pre_net import Pre_Net
from .input_mask import create_masks, create_combined_mask, create_padding_mask

BEAM_SIZE = 2

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
                 num_layers_enc=8, 
                 num_layers_dec=4, 
                 d_model=256, 
                 num_heads=4, 
                 dff=1024, 
                 dropout=0.1 ):
        super(SpeechTransformer, self).__init__()

        self.pre_net = Pre_Net(num_M, n, c)
        self.transformer = Transformer(num_layers_enc, num_layers_dec, d_model, num_heads, dff, target_vocab_size, dropout)

    def forward(self, inputs, targets, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        out = self.pre_net(inputs)
        out = self.transformer(inputs=(out, targets),
                               enc_padding_mask=enc_padding_mask,
                               look_ahead_mask=look_ahead_mask,
                               dec_padding_mask=dec_padding_mask )
        
        return out

    def beam_search_decoding(self, inp, sequence_sizes, start_token, end_token):
        """
        Perform beam search decoding within the SpeechTransformer model.
        Args:
            inp (torch.Tensor): Input tensor (audio features).
            max_len (int): Maximum length of the generated sequence.
            start_token (int): Start token index.
            end_token (int): End token index.
        Returns:
            List[torch.Tensor]: Decoded sequences for each batch.
        """
        batch_size = inp.size(0)
        sequences = [[(torch.tensor([start_token], device=inp.device), 0.0)] for _ in range(batch_size)]

        # Create encoder padding mask and process encoder output
        enc_padding_mask, _, _ = create_masks(inp, torch.zeros((batch_size, 1), dtype=torch.int64, device=inp.device))
        enc_output = self.pre_net(inp)
        enc_output = self.transformer.encoder(x=enc_output, mask=enc_padding_mask)

        max_len = max(sequence_sizes)
        for i in range(max_len):
            all_candidates = []
            for batch_idx, batch_seq in enumerate(sequences):
                for seq, score in batch_seq:
                    if (seq[-1] == end_token) or (i >= sequence_sizes[batch_idx]):
                        all_candidates.append((seq, score))
                        continue

                    # Prepare decoder input and masks for this sequence
                    tar = seq.unsqueeze(0)  # Add batch dimension
                    combined_mask = create_combined_mask(tar)
                    dec_padding_mask = create_padding_mask(inp[batch_idx:batch_idx+1, :enc_output.size(1), 0, 0], (1, 1, tar.size(1), enc_output.size(1)))

                    with torch.no_grad():
                        dec_output = self.transformer.decoder(
                            x=tar,
                            enc_output=enc_output[batch_idx:batch_idx+1],
                            look_ahead_mask=combined_mask,
                            padding_mask=dec_padding_mask
                        )

                        log_probs = torch.log_softmax(dec_output[:, -1, :], dim=-1)
                        topk_probs, topk_indices = torch.topk(log_probs, BEAM_SIZE, dim=-1)

                        for i in range(BEAM_SIZE):
                            candidate = (
                                torch.cat([seq, topk_indices[0, i].unsqueeze(0)]),
                                score - topk_probs[0, i].item()
                            )
                            all_candidates.append(candidate)

                # Sort and select top BEAM_SIZE sequences
                ordered = sorted(all_candidates, key=lambda x: x[1] / len(x[0]))
                sequences[batch_idx] = ordered[:BEAM_SIZE]

        # Select the best sequence for each batch
        final_sequences = [sorted(batch_seq, key=lambda x: x[1] / len(x[0]))[0][0] for batch_seq in sequences]
        return final_sequences
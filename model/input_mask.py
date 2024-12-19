import torch

def create_padding_mask(seq, target_shape):
    """
    Creates a padding mask for sequences.
    Args:
        seq (torch.Tensor): Input tensor with padding values (0 for padding).
        target_shape (tuple): Shape to which the padding mask will be resized.
    Returns:
        torch.Tensor: Resized padding mask for scaled_attention_logits.
    """
    mask = (seq == 0).unsqueeze(1).unsqueeze(2).float()  # Initial shape: (batch_size, 1, 1, seq_len)
    return mask.expand(-1, -1, target_shape[2], -1)  # Expand to match attention logits' shape

def create_look_ahead_mask(size):
    """
    Creates a look-ahead mask for decoding.
    Args:
        size (int): Length of the target sequence.
    Returns:
        torch.Tensor: Look-ahead mask with shape (seq_len, seq_len).
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.float()  # Shape: (seq_len, seq_len)

def create_combined_mask(tar):
    """
    Combines the look-ahead mask and padding mask for the decoder self-attention.
    Args:
        tar (torch.Tensor): Target sequence tensor.
    Returns:
        torch.Tensor: Combined mask for decoder with shape (batch_size, 1, seq_len, seq_len).
    """
    look_ahead_mask = create_look_ahead_mask(tar.size(1)).to(tar.device)  # Shape: (seq_len, seq_len)
    dec_target_padding_mask = (tar == 0).unsqueeze(1).unsqueeze(2).float()  # Padding mask: (batch_size, 1, 1, seq_len)
    combined_mask = torch.max(dec_target_padding_mask.squeeze(1), look_ahead_mask)  # Combine
    return combined_mask.unsqueeze(1)  # Shape: (batch_size, 1, seq_len, seq_len)

def create_masks(inp, tar):
    """
    Creates all masks required for the Transformer model.
    Args:
        inp (torch.Tensor): Input tensor for the encoder, shape [batch_size, audio_length, ...].
        tar (torch.Tensor): Target tensor for the decoder, shape [batch_size, text_length].
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Encoder padding mask, combined mask, decoder padding mask.
    """
    audio_length = int(torch.ceil(torch.tensor(inp.size(1) / 4.0)))
    batch_size, text_length = tar.size()
    enc_padding_mask = create_padding_mask(inp[:, :audio_length, 0, 0], (batch_size, 1, audio_length, audio_length))  # Shape: (batch_size, 1, audio_length, audio_length)
    combined_mask = create_combined_mask(tar)  # Shape: (batch_size, 1, text_length, text_length)
    dec_padding_mask = create_padding_mask(inp[:, :audio_length, 0, 0], (batch_size, 1, text_length, audio_length))  # Shape: (batch_size, 1, text_length, audio_length)

    return enc_padding_mask, combined_mask, dec_padding_mask

'''
I think the masks that uses 'audio_length' are not implemented correctly, 
But I don't know how to do it considering they are created before the beggining
of the Pre_Net. Maybe if they were created after audio went throught the 
convolutions, it would be easier to implement.
'''
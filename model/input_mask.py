import torch

PAD = -1

def create_padding_mask(seq, device):
    return (seq == 0).float().unsqueeze(1).unsqueeze(2).to(device)

def create_look_ahead_mask(size, device):
    mask = torch.triu(torch.ones(size, size), diagonal=1).float().to(device)
    return mask

def create_masks(inp, tar):
    device = inp.device
    enc_padding_mask = create_padding_mask(inp, device)
    dec_padding_mask = create_padding_mask(inp, device)
    look_ahead_mask = create_look_ahead_mask(tar.size(1), device)
    dec_target_padding_mask = create_padding_mask(tar, device)
    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

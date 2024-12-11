import torch

def create_padding_mask(seq):
    return (seq == 0).unsqueeze(1).unsqueeze(2).float()

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.float()

def create_combined_mask(tar):
    look_ahead_mask = create_look_ahead_mask(tar.size(1)).to(tar.device)
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = torch.max(dec_target_padding_mask.squeeze(1), look_ahead_mask)
    combined_mask = combined_mask.unsqueeze(1)

    return combined_mask

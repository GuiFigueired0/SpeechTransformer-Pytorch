import torch

def get_angles(pos, i, d_model):
    """
    Computes the angle rates for positional encoding.
    """
    angle_rates = 1 / (10000 ** (2 * (i // 2) / d_model))
    return pos * angle_rates

def positional_encoding(position, d_model, device):
    """
    Generates sinusoidal positional encoding.
    Args:
        position: Number of positions (max sequence length).
        d_model: Depth of the model.
        device: Device to place the tensor (e.g., 'cpu' or 'cuda:0').
    Returns:
        Tensor of shape (1, position, d_model) with positional encodings.
    """
    angle_rads = get_angles(
        torch.arange(position, device=device).unsqueeze(1),
        torch.arange(d_model, device=device).unsqueeze(0),
        d_model
    )
    
    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])
    pos_encoding = torch.cat([sines, cosines], dim=-1).unsqueeze(0)
    
    return pos_encoding.float()

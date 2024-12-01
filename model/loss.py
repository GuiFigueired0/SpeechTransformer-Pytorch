import torch
import torch.nn.functional as F

def label_smoothing(inputs, epsilon=0.1):
    K = inputs.size(-1)
    return ((1 - epsilon) * inputs) + (epsilon / K)

def label_smoothing_loss(real, pred, vocab_size, epsilon=0.1):
    real_one_hot = F.one_hot(real, num_classes=vocab_size).float()
    real_smoothed = label_smoothing(real_one_hot, epsilon)
    loss = F.cross_entropy(pred, real_smoothed, reduction='none')

    mask = real != 0
    loss = loss * mask.float()
    return loss.mean()

def loss_fn(real, pred):
    loss = F.cross_entropy(pred, real, reduction='none')
    mask = real != 0
    loss = loss * mask.float()
    return loss.mean()
import torch
import torch.nn as nn
import torch.nn.functional as F

def label_smoothing(inputs, epsilon=0.1):
    vocab_size = inputs.size(-1) # Vocabulary size (last dimension)
    smoothed_inputs = (1 - epsilon) * inputs + (epsilon / vocab_size)
    return smoothed_inputs

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, epsilon=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.epsilon = epsilon

    def forward(self, real, pred):
        real_one_hot = F.one_hot(real, num_classes=self.vocab_size).float() # Shape: [batch_size, sequence_length, vocab_size]
        real_smoothed = label_smoothing(real_one_hot, self.epsilon)

        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(real_smoothed * log_probs).sum(dim=-1) # Sum over vocab dimension: [batch_size, sequence_length]

        mask = real != 0  
        loss = loss * mask.float()  

        return loss.sum() / mask.sum()

if __name__ == "__main__":
    vocab_size = 5
    batch_size = 3
    seq_len = 4
    epsilon = 0.1

    pred = torch.randn(batch_size, seq_len, vocab_size)  
    real = torch.tensor([[1, 2, 0, 0], [2, 3, 1, 0], [0, 1, 2, 3]])

    loss_fn = LabelSmoothingLoss(vocab_size=vocab_size, epsilon=epsilon)

    loss = loss_fn(real, pred)
    print("Loss:", loss.item())

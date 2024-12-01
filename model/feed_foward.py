import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
from torch import nn
from torch.nn import functional as F
import torch

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.linear1 = nn.Linear(config.hidden_size, config.ffn_hidden)
        self.linear2 = nn.Linear(config.ffn_hidden, config.hidden_size)
        self.dropout = nn.Dropout(config.drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
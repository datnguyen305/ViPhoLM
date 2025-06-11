import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentSelector(nn.Module):
    """Bottom-up content selector module"""
    def __init__(self, hidden_size):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, encoder_outputs):
        """Given encoder outputs, predict the importance of each token"""
        return torch.sigmoid(self.scorer(encoder_outputs))

from torch import nn
from models.transformer_pointer.layers.modules.utils import clones
import torch 
from models.transformer_pointer.layers.modules.ocr_layer import OCREncoderLayer
from models.transformer_pointer.embedding.positional_embedding import PositionalEncoding
class OCREncoder(nn.Module):
    def __init__(self, config, word_embed = None):
        
        super().__init__()        
        self.config = config
        self.word_embed = word_embed
        self.layers = clones(
            OCREncoderLayer(self.config.head, \
                self.config.d_model, self.config.d_kv, \
                self.config.d_ff) \
                , 3 
            )
        self.norm = nn.LayerNorm(self.config.d_model)

    def forward(self, input: torch.Tensor, \
        mask: torch.Tensor, PE):
        """
        input: (B, S)
        mask: (B, 1, 1, S)
        
        Returns: 
            x: (B, S, H)
            break_probs: (B, S, S)
        """
        break_probs = []
        x = self.word_embed(input)
        # x: (B, S, H)
        x = PE(x)
        # x: (B, S, H)
        
        group_prob = 0.0
        for layer in self.layers:
            x, group_prob, break_prob = layer(x, mask, group_prob)
            break_probs.append(break_prob)

        x = self.norm(x)
        break_probs = torch.stack(break_probs, dim=1)

        return x, break_probs
from torch import nn 
from models.enhanced_attn_transformer.layers.standard_encoder_layer import StandardEncoderLayer

class StandardTransformerEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([StandardEncoderLayer(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

from torch import nn 
from models.enhanced_attn_transformer.layers.positionwise_feed_forward import PositionwiseFeedForward
from models.enhanced_attn_transformer.utils.clone import clones
from models.enhanced_attn_transformer.layers.sub_layer_connection import SublayerConnection

class StandardEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.d_model, config.head, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

    def forward(self, x, mask):
        # mask ở đây là padding mask cho các câu (B, S)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, key_padding_mask=mask)[0])
        return self.sublayer[1](x, self.feed_forward)
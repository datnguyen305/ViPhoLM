from vocabs.hierarchy_vocab import Hierarchy_Vocab
from torch import nn 
from models.enhanced_attn_transformer.layers.group_attention import GroupAttention
from models.enhanced_attn_transformer.layers.scaled_dot_product_attention import ScaledDotProductAttention
from models.enhanced_attn_transformer.layers.positionwise_feed_forward import PositionwiseFeedForward
from models.enhanced_attn_transformer.utils.clone import clones
from models.enhanced_attn_transformer.layers.sub_layer_connection import SublayerConnection

class EncoderLayer(nn.Module):
    def __init__(self, config, vocab: Hierarchy_Vocab):
        super().__init__()
        # Attention
        self.group_attn = GroupAttention(config)
        self.self_attn = ScaledDotProductAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)
        self.size = config.d_model

    def forward(self, x, mask, group_prob):
        group_prob, break_prob = self.group_attn(x, mask, group_prob)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob, mask))
        
        return self.sublayer[1](x, self.feed_forward), group_prob, break_prob
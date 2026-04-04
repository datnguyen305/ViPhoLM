from models.transformer_pointer.layers.modules.group_attn import GroupAttention
from models.transformer_pointer.layers.modules.multi_head_attn import ScaledDotProductAttention
from models.transformer_pointer.layers.modules.pos_ffw import PositionwiseFeedForward
from models.transformer_pointer.layers.modules.sub_layer_connection import SublayerConnection
from torch import nn
from models.transformer_pointer.layers.modules.utils import clones

class OCREncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, head, d_model, d_kv, d_ff, dropout=0.1):
        super().__init__()
        self.group_attn = GroupAttention(head, d_model)
        self.self_attn = ScaledDotProductAttention(head, d_model, d_kv)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.size = d_model

    def forward(self, x, mask, group_prob):
        group_prob, break_prob = self.group_attn(x, mask, group_prob)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob, mask))
        
        return self.sublayer[1](x, self.feed_forward), group_prob, break_prob
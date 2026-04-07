import torch 
from torch import nn 
from models.unilm.layers.positionwise_feed_forward import PositionwiseFeedForward
from models.unilm.utils.clone import clones
from models.unilm.layers.sub_layer_connection import SublayerConnection
from models.unilm.layers.scaled_dot_product_attention import ScaledDotProductAttention
from vocabs.unilm_vocab import UniLM_Vocab

class EncoderLayer(nn.Module):
    def __init__(self, config, vocab: UniLM_Vocab):
        super().__init__()
        self.self_attn = ScaledDotProductAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

    def forward(self, x, mask):
    
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)[0])
        # x: (B, S, d_model)

        return self.sublayer[1](x, self.feed_forward)
    
        # output: (B, S, d_model)
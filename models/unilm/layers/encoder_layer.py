import torch 
from torch import nn 
from models.unilm.layers.positionwise_feed_forward import PositionwiseFeedForward
from models.unilm.utils.clone import clones
from models.unilm.layers.sub_layer_connection import SublayerConnection
from vocabs.unilm_vocab import UniLM_Vocab

class EncoderLayer(nn.Module):
    def __init__(self, config, vocab: UniLM_Vocab):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.d_model, config.head, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

    def forward(self, x, mask):
    
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, key_padding_mask=mask)[0])
        # x: (B, S, d_model)

        return self.sublayer[1](x, self.feed_forward)
    
        # output: (B, S, d_model)
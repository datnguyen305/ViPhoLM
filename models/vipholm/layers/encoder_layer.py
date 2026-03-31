import torch 
from torch import nn 
from models.vipholm.layers.positionwise_feed_forward import PositionwiseFeedForward
from models.vipholm.utils.clone import clones
from models.vipholm.layers.sub_layer_connection import SublayerConnection
from models.vipholm.layers.longformer_self_attention import LongformerSelfAttention
from vocabs.viword_vocab import ViWordVocab

class EncoderLayer(nn.Module):
    def __init__(self, config, vocab: ViWordVocab, layer_id: int):
        super().__init__()


        self.self_attn = LongformerSelfAttention(
            config, layer_id)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, attention_mask=mask)[0])
        # x: (B, S, d_model)

        return self.sublayer[1](x, self.feed_forward)
    
        # output: (B, S, d_model)
import torch 
from torch import nn 
from models.transformer_phoneme_longformer.layers.positionwise_feed_forward import PositionwiseFeedForward
from models.transformer_phoneme_longformer.utils.clone import clones
from models.transformer_phoneme_longformer.layers.sub_layer_connection import SublayerConnection
from models.transformer_phoneme_longformer.layers.longformer_self_attention import LongformerSelfAttention
from vocabs.viword_vocab import ViWordVocab

class EncoderLayer(nn.Module):
    def __init__(self, config, vocab: ViWordVocab):
        super().__init__()


        self.self_attn = LongformerSelfAttention(
            config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

    def forward(self, x, mask, id):
    
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, key_padding_mask=mask, layer_id = id)[0])
        # x: (B, S, d_model)

        return self.sublayer[1](x, self.feed_forward)
    
        # output: (B, S, d_model)
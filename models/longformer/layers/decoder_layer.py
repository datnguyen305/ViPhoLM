from torch import nn 
import torch
from models.longformer.layers.positionwise_feed_forward import PositionwiseFeedForward
from models.longformer.utils.clone import clones
from models.longformer.layers.sub_layer_connection import SublayerConnection

class DecoderLayer(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.num_features = 1
        self.self_attn = nn.MultiheadAttention(config.d_model, config.head, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.head, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 3)

        self.embedding = clones(nn.Embedding(vocab.vocab_size, config.d_model), self.num_features)
        self.linear = nn.Linear(config.d_model * 3, config.d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, memory, tgt_causal_mask, tgt_padding_mask, memory_padding_mask):
        
        """
        x: (Batch, Seq_Len, Dim)
        tgt_causal_mask: (Seq_Len, Seq_Len) - Mask che tương lai (Float -inf)
        tgt_padding_mask: (Batch, Seq_Len) - Mask che padding target (Bool True/False)
        memory_padding_mask: (Batch, Source_Len) - Mask che padding source (Bool True/False)
        """
        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, 
                                                         attn_mask=tgt_causal_mask,
                                                         key_padding_mask=tgt_padding_mask)[0])
        
        x = self.sublayer[1](x, lambda x: self.cross_attn(query=x, 
                                                         key=memory, 
                                                         value=memory, 
                                                         key_padding_mask=memory_padding_mask)[0])
        
        x = self.sublayer[2](x, self.feed_forward)
        
        return x



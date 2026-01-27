from torch import nn
from vocabs.vocab import Vocab
from ..layer.encoder_layers import TransformerEncoderLayer


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config, vocab) for _ in range(config.n_layers)
        ])

    def forward(self, src, src_mask=None, src_causal_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, src_causal_mask=src_causal_mask)
        return output
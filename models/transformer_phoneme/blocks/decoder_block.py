from torch import nn
from vocabs.vocab import Vocab
from ..layer.decoder_layers import TransformerDecoderLayer
class TransformerDecoderBlock(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config, vocab) for _ in range(config.n_layers)
        ])

    def forward(self, trg, memory, trg_mask=None, trg_causal_mask=None, src_mask=None):
        output = trg
        for layer in self.layers:
            output = layer(output, memory, trg_mask=trg_mask, trg_causal_mask=trg_causal_mask, src_mask=src_mask)
        return output
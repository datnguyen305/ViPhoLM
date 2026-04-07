from torch import nn
from vocabs.unilm_vocab import UniLM_Vocab
from models.unilm.layers.encoder_layer import EncoderLayer
from models.unilm.utils.clone import clones

class TransformerBlock(nn.Module):
    def __init__(self, config, vocab: UniLM_Vocab):
        super().__init__()
        self.layers = clones(EncoderLayer(config, vocab), config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
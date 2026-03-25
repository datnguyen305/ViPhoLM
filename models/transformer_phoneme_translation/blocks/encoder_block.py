from torch import nn
from vocabs.vocab_translation import MTVocab
from models.transformer_phoneme_translation.layers.encoder_layer import EncoderLayer
from models.transformer_phoneme_translation.utils.clone import clones


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config, vocab: MTVocab):
        super().__init__()
        self.layers = clones(EncoderLayer(config, vocab), config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
from torch import nn
from vocabs.viword_vocab import ViWordVocab 
from models.transformer_phoneme.layers.encoder_layer import EncoderLayer
from models.transformer_phoneme.utils.clone import clones


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.num_features = 3
        self.embedding = clones(nn.Embedding(vocab.vocab_size, config.d_model), self.num_features)
        self.layers = clones(EncoderLayer(config), config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
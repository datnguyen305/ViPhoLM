from torch import nn

from models.transformer_seal.blocks.encoder_layer import EncoderLayer
from models.transformer_seal.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, config, vocab):
        super().__init__()
        self.emb = TransformerEmbedding(d_model = config.d_model,
                                        max_len = config.max_len,
                                        vocab_size = vocab.vocab_size,
                                        drop_prob = config.drop_prob,
                                        device = config.device)

        self.layers = nn.ModuleList([EncoderLayer(d_model = config.d_model,
                                                  ffn_hidden = config.ffn_hidden,
                                                  n_head = config.n_head,
                                                  drop_prob = config.drop_prob,
                                                  segment_size = getattr(config, 'segment_size', 512))
                                     for _ in range(config.n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
import torch
from torch import nn

from models.transformer_seal.blocks.decoder_layer import DecoderLayer
from models.transformer_seal.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.emb = TransformerEmbedding(d_model = config.d_model,
                                        drop_prob = config.drop_prob,
                                        max_len = config.max_len,
                                        vocab_size = vocab.vocab_size,
                                        device = config.device)

        self.layers = nn.ModuleList([DecoderLayer(d_model = config.d_model,
                                                  ffn_hidden = config.ffn_hidden,
                                                  n_head = config.n_head,
                                                  drop_prob = config.drop_prob,
                                                  segment_size = getattr(config, 'segment_size', 512))
                                     for _ in range(config.n_layers)])

        self.linear = nn.Linear(config.d_model, vocab.vocab_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output
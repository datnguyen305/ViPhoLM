import torch
from torch import nn

from models.transformer.blocks.decoder_layer import DecoderLayer
from models.transformer.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size= vocab.vocab_size,
                                        d_model=config.d_model,
                                        max_len=config.max_len,
                                        drop_prob=config.drop_prob,
                                        device=config.device)

        self.layers = nn.ModuleList([DecoderLayer(d_model = config.d_model,
                                                  ffn_hidden = config.ffn_hidden,
                                                  n_head = config.n_head,
                                                  drop_prob = config.drop_prob)
                                     for _ in range(config.n_layers)])

        self.linear = nn.Linear(config.d_model, vocab.vocab_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg) # # -> [batch, trg_len, d_model]

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg) # # -> [batch, src_len, vocab_size]
        return output
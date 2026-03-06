from torch import nn  
from models.enhanced_attn_transformer.utils.clone import clones
import torch 
from vocabs.hierarchy_vocab import Hierarchy_Vocab
from models.enhanced_attn_transformer.embedding.positional_embedding import PositionalEncoding
from models.enhanced_attn_transformer.layers.encoder_layer import EncoderLayer


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config, vocab: Hierarchy_Vocab):
        super().__init__()
        self.word_embed = nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=vocab.pad_idx)
        self.layers = clones(EncoderLayer(config, vocab), 3)
        self.norm = nn.LayerNorm(config.d_model)

        self.pos_embed = PositionalEncoding(config.d_model)

    def forward(self, inputs, mask):
        x = self.word_embed(inputs)
        x = self.pos_embed(x)

        break_probs = []
        group_prob = 0.

        for layer in self.layers:
            x, group_prob, break_prob = layer(x, mask,group_prob)
            break_probs.append(break_prob)

        x = self.norm(x)
        break_probs = torch.stack(break_probs, dim=1)

        return x, break_probs
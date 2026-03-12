from torch import nn 
from models.transformer_phoneme.utils.clone import clones
from models.transformer_phoneme.embedding.positional_embedding import PositionalEncoding
from models.transformer_phoneme.layers.decoder_layer import DecoderLayer 

class TransformerDecoderBlock(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.embedding = nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=vocab.pad_idx)
        self.pos_embed = PositionalEncoding(config.d_model)
        self.layers = clones(DecoderLayer(config, vocab), config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, trg, memory, tgt_causal_mask, tgt_padding_mask, memory_padding_mask):
        x = self.embedding(trg)
        x = self.pos_embed(x)
        
        for layer in self.layers:
            x = layer(x, memory, tgt_causal_mask, tgt_padding_mask, memory_padding_mask)
            
        return self.norm(x)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.bottom_up.selector import ContentSelector

class Encoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx=vocab.pad_idx)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, bidirectional=config.bidirectional,
                            num_layers=config.layer_dim, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.content_selector = ContentSelector(config.hidden_size)

    def forward(self, input, input_lengths):
        embedded = self.dropout(self.embedding(input))
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, states = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        copy_probs = self.content_selector(outputs)
        return outputs, states, copy_probs

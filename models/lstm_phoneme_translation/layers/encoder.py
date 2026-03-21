import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.vocab_translation import MTVocab

class Encoder(nn.Module):
    def __init__(self, config, vocab: MTVocab):
        super().__init__()
        self.num_features = 1
        self.embedding = nn.Embedding(
            vocab.english_vocab_size, config.hidden_size
        )
        self.lstm = nn.LSTM(
            config.hidden_size * self.num_features,
            config.hidden_size,
            bidirectional=False,
            num_layers=config.layer_dim, # 3 lstm
            batch_first=True, 
            dropout=config.dropout
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        # input: (B, S)
        print(input.shape)
        B, S = input.size()
        embed = self.embedding(input)
        # embed: (batch_size, seq_len, hidden_size)

        output, state = self.lstm(embed)
        # output : (batch_size, seq_len, hidden_size)
        # states: (h_n, c_n) final_hidden_state: (num_layers, batch_size, hidden_size)

        return output, state
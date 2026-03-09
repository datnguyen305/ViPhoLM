import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.viword_vocab import ViWordVocab
from builders.model_builder import META_ARCHITECTURE

class Encoder(nn.Module):
    def __init__(self, config, vocab: ViWordVocab):
        super().__init__()
        self.num_features = 3
        self.embedding = nn.ModuleList([
            nn.Embedding(vocab.vocab_size, config.hidden_size) 
            for _ in range(self.num_features)
        ])
        self.lstm = nn.LSTM(
            config.hidden_size * self.num_features,
            config.hidden_size * 3,
            bidirectional=False,
            num_layers=config.layer_dim, # 3 lstm
            batch_first=True, 
            dropout=config.dropout
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        B, S, _  = input.size()
        embeds = []
        for i in range(self.num_features):
            embeds.append(self.dropout(self.embedding[i](input[:, :, i])))
            # embeds: (batch_size, seq_len, hidden_size) * 3
        embedded = torch.cat(embeds, dim=-1)
        # embedded: (batch_size, seq_len, hidden_size * 3)

        output, state = self.lstm(embedded)
        # output : (batch_size, seq_len, hidden_size * 3)
        # states: (h_n, c_n) final_hidden_state: (num_layers, batch_size, hidden_size * 3)

        return output, state
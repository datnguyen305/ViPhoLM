import torch.nn as nn
from vocabs.viword_vocab import ViWordVocab

class Encoder(nn.Module):
    def __init__(self, config, vocab: ViWordVocab):
        super().__init__()
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.emb_proj = nn.Linear(
            in_features=config.hidden_size*3,
            out_features=config.hidden_size
        )
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            bidirectional=False,
            num_layers=config.layer_dim, # 3 lstm
            batch_first=True, 
            dropout=config.dropout
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        B, L, _, _ = embedded.shape
        embedded = embedded.reshape(B, L, -1)
        embedded = self.emb_proj(embedded)

        output, state = self.lstm(embedded)
        # output : (batch_size, seq_len, hidden_size)
        # states: (h_n, c_n) final_hidden_state: (num_layers, batch_size, hidden_size)

        return output, state
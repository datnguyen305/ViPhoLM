import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.viword_vocab import ViWordVocab
from models.lstm_phoneme.layers.feed_forward import FeedForward

class Decoder(nn.Module):
    def __init__(self, config, vocab: ViWordVocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.num_features = 3
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.emb_proj = nn.Linear(
            in_features=config.hidden_size*3,
            out_features=config.hidden_size
        )

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            bidirectional=False,
            num_layers=config.layer_dim, # 3 lstm 
            batch_first=True, 
            dropout=config.dropout
        )
        self.fflayers = nn.ModuleList(
            FeedForward(config) \
            for _ in range(self.num_features)
        )
        self.outs = nn.ModuleList(
            nn.Linear(config.hidden_size, vocab.vocab_size)
            for _ in range(self.num_features)
        )
    def forward(self, encoder_outputs: torch.Tensor, encoder_states: torch.Tensor, target_tensor: torch.Tensor):
        batch_size = encoder_outputs.size(0)

        # Initiate decoder's input [<BOS>, <BOS>, <BOS>]
        decoder_input = torch.empty(batch_size, 1, self.num_features, dtype=torch.long, device=encoder_outputs.device)
        decoder_input.fill_(self.vocab.bos_idx)
        # decoder_input: (batch_size, 1, 3)

        decoder_hidden, decoder_memory = encoder_states
        # decoder_hidden: (num_layers, batch_size, hidden_size * 3)
        # decoder_memory: (num_layers, batch_size, hidden_size * 3)

        decoder_outputs = []
        target_len = target_tensor.shape[1]

        for i in range(target_len):
            decoder_output, (decoder_hidden, decoder_memory) = self.forward_step(decoder_input, (decoder_hidden, decoder_memory))
            # decoder_output: (batch_size, 1, vocab_size, 3)

            decoder_output.reshape(batch_size, 1, -1)
            # decoder_output: (batch_size, 1, vocab_size * 3)

            decoder_outputs.append(decoder_output)

            # Teacher forcing: Feed the target as the next input
            decoder_input = target_tensor[:, i, :].unsqueeze(1) # Teacher forcing

        # decoder_output: (batch_size, 1, vocab_size * 3)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs: (batch_size, target_len, vocab_size * 3) 
        decoder_outputs.reshape(batch_size, target_len, self.vocab.vocab_size, -1)
        # decoder_outputs: (batch_size, target_len, vocab_size, 3)
        return decoder_outputs, (decoder_hidden, decoder_memory)

    def forward_step(self, input, states):
        embedded = self.embedding(input)
        B, L, _ = input.shape
        embedded = embedded.reshape(B, L, -1)
        # embedded: (batch_size, 1, hidden_size * 3)
        embedded = self.emb_proj(embedded)
        # embedded: (batch_size, 1, hidden_size)

        output, (hidden, memory) = self.lstm(embedded, states)
        # output: (batch_size, 1, hidden_size)

        ff_outputs = []
        for fflayer in self.fflayers:
            ff_outputs.append(fflayer(output))

        outputs = []
        for out_layer, ff_output in zip(self.outs, ff_outputs):
            outputs.append(out_layer(ff_output))
        # outputs: (batch_size, 1, vocab_size) * 3

        outputs = torch.stack(outputs, -2)
        # outputs: (batch_size, 1, 3, vocab_size)

        return outputs, (hidden, memory)

import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.vocab_translation import MTVocab
from models.lstm_phoneme.layers.feed_forward import FeedForward
from builders.model_builder import META_ARCHITECTURE

class Decoder(nn.Module):
    def __init__(self, config, vocab: MTVocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.num_features = 3
        self.embedding = nn.ModuleList([
            nn.Embedding(vocab.vietnamese_vocab_size, config.hidden_size) 
            for _ in range(self.num_features)
        ])
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.linear_prj = nn.Linear(config.hidden_size, config.hidden_size * 3)


        self.lstm = nn.LSTM(
            config.hidden_size * self.num_features,
            config.hidden_size * self.num_features,
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
            nn.Linear(config.hidden_size, vocab.vietnamese_vocab_size)
            for _ in range(self.num_features)
        )
    def forward(self, encoder_outputs: torch.Tensor, encoder_states: torch.Tensor, target_tensor: torch.Tensor):
        
        batch_size = encoder_outputs.size(0)

        # Initiate decoder's input [<BOS>, <PAD>, <PAD>]
        decoder_input = torch.empty(batch_size, 1, self.num_features, dtype=torch.long, device=encoder_outputs.device)
        for i in range(self.num_features):
            if i == 0: 
                decoder_input[:, :, i].fill_(self.vocab.bos_idx)
            else: 
                decoder_input[:, :, i].fill_(self.vocab.pad_idx)
        # decoder_input: (batch_size, 1, 3)

        decoder_hidden, decoder_memory = encoder_states
        # decoder_hidden: (num_layers, batch_size, hidden_size)
        # decoder_memory: (num_layers, batch_size, hidden_size)

        decoder_hidden = self.linear_prj(decoder_hidden)
        decoder_memory = self.linear_prj(decoder_memory)
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
        decoder_outputs.reshape(batch_size, target_len, self.vocab.vietnamese_vocab_size, -1)
        # decoder_outputs: (batch_size, target_len, vocab_size, 3)
        return decoder_outputs, (decoder_hidden, decoder_memory)

    def forward_step(self, input, states):
        B = input.shape[0]
        hidden_size = self.hidden_size
        embeds = []
        for i in range(self.num_features):
            embeds.append(self.dropout(self.embedding[i](input[:, :, i])))
        embedded = torch.cat(embeds, dim=-1)
        # embedded: (batch_size, 1, hidden_size * 3)

        output, (hidden, memory) = self.lstm(embedded, states)
        # output: (batch_size, 1, hidden_size * 3)

        output = output.reshape(B, 1, hidden_size, 3)
        # output: (batch_size, 1, hidden_size, 3)

        ff_outputs = []
        for i in range(self.num_features):
            ff_outputs.append(self.fflayers[i](output[:, :, :, i]))
        # ff_outputs: (batch_size, 1, hidden_size) * 3

        outputs = []
        for i in range(self.num_features):
            outputs.append(self.outs[i](ff_outputs[i]))
        # outputs: (batch_size, 1, vocab_size) * 3

        batch_size = outputs
        outputs = torch.stack(outputs, -1)
        # outputs: (batch_size, 1, vocab_size, 3)

        return outputs, (hidden, memory)
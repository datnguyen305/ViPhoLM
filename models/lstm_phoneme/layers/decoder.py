import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.viword_vocab import ViWordVocab
from models.lstm_phoneme.layers.feed_forward import FeedForward
from builders.model_builder import META_ARCHITECTURE

class Decoder(nn.Module):
    def __init__(self, config, vocab: ViWordVocab):
        super().__init__()
        self.num_features = 3
        self.embedding = nn.ModuleList([
            nn.Embedding(vocab.vocab_size, config.hidden_size) 
            for _ in range(self.num_features)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            config.hidden_size * self.num_features,
            config.hidden_size * self.num_features,
            bidirectional=False,
            num_layers=config.layer_dim, # 3 lstm 
            batch_first=True, 
            dropout=config.dropout
        )



    def forward(self, encoder_outputs: torch.Tensor, encoder_states: torch.Tensor, target_tensor: torch.Tensor):
        batch_size = encoder_outputs.size(0)
        # Initiate decoder's input [<BOS>, <PAD>, <PAD>, <PAD>]
        decoder_input = torch.empty(batch_size, 1, self.num_features, dtype=torch.long, device=encoder_outputs.device)
        for i in range(self.num_features):
            if i == 0: 
                decoder_input[:, :, i].fill_(self.vocab.bos_idx)
            else: 
                decoder_input[:, :, i].fill_(self.vocab.pad_idx)
        # decoder_input: (batch_size, 1, 4)

        decoder_hidden, decoder_memory = encoder_states
        decoder_outputs = []
        target_len = target_tensor.shape[1]
        # target_tensor: (batch_size, seq_len, 4)
        # target_tensor [: , i] (batch_size, 4)
        for i in range(target_len):
            decoder_output, (decoder_hidden, decoder_memory) = self.forward_step(decoder_input, (decoder_hidden, decoder_memory))
            # decoder_output: (B, 1, 4, vocab_size)
            decoder_outputs.append(decoder_output)

            # Teacher forcing: Feed the target as the next input
            decoder_input = target_tensor[:, i, :].unsqueeze(1) # Teacher forcing

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs: (B, S, 4, vocab_size)
        
        return decoder_outputs, (decoder_hidden, decoder_memory)

    def forward_step(self, input, states):
        embeds = []
        for i in range(self.num_features):
            embeds.append(self.dropout(self.embedding[i](input[:, :, i])))
        embedded = torch.cat(embeds, dim=-1)
        # embedded: (batch_size, 1, hidden_size * 4)
        output, (hidden, memory) = self.lstm(embedded, states)
        # output: (batch_size, 1, hidden_size)
        onset_out = self.fc_onset(output)
        medial_out = self.fc_medial(output)
        nucleus_out = self.fc_nucleus(output)
        coda_out = self.fc_coda(output)

        # *_out :(batch_size, 1, vocab_size)
        output = torch.stack([onset_out, medial_out, nucleus_out, coda_out], dim=2)
        # output: (B, 1, 4, vocab_size)

        return output, (hidden, memory)
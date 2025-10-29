import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class Encoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.bidirectional = config.bidirectional
        self.hidden_size = config.hidden_size
        lstm_hidden_size = config.hidden_size // 2 if self.bidirectional else config.hidden_size
        self.layer_dim = config.layer_dim

        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.lstm = nn.LSTM(
            config.hidden_size,
            lstm_hidden_size,
            bidirectional=config.bidirectional,
            num_layers=config.layer_dim,
            batch_first=True,
            dropout=config.dropout
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, states = self.lstm(embedded)

        if self.bidirectional:
            # Concatenate the forward and backward states
            h, c = states

            h = torch.cat((h[0:h.size(0):2], h[1:h.size(0):2]), dim=-1)
            c = torch.cat((c[0:c.size(0):2], c[1:c.size(0):2]), dim=-1)

            states = (h, c)

        return output, states
    
class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.bidirectional = config.bidirectional
        self.hidden_size = config.hidden_size
        lstm_hidden_size = config.hidden_size // 2 if self.bidirectional else config.hidden_size

        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)

        self.lstm = nn.LSTM(
            config.hidden_size,
            lstm_hidden_size,
            bidirectional=config.bidirectional,
            num_layers=config.layer_dim,
            batch_first=True,
            dropout=config.dropout
        )

        self.out = nn.Linear(config.hidden_size, vocab.vocab_size)

    def forward(self, encoder_outputs: torch.Tensor, encoder_states: torch.Tensor, target_tensor: torch.Tensor):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(self.vocab.bos_idx)
        
        decoder_hidden, decoder_memory = encoder_states
        decoder_outputs = []
        target_len = target_tensor.shape[-1]

        for i in range(target_len):
            decoder_output, (decoder_hidden, decoder_memory) = self.forward_step(decoder_input, (decoder_hidden, decoder_memory))
            decoder_outputs.append(decoder_output)
            # teacher forcing 
            decoder_input = target_tensor[:, i].unsqueeze(1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, (decoder_hidden, decoder_memory)

    def forward_step(self, input, states):
        output = self.embedding(input)
        output = F.relu(output)
        output, (hidden, memory) = self.lstm(output, states)
        output = self.out(output)
        return output, (hidden, memory)

@META_ARCHITECTURE.register()
class BiLSTM_Model(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2
        self.d_model = config.d_model

        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)
        self.pad_idx = vocab.pad_idx

        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        encoder_outs, hidden_states = self.encoder(x)

        outs, _ = self.decoder(encoder_outs, hidden_states, labels)

        loss = self.loss(outs.reshape(-1, self.vocab.vocab_size), labels.reshape(-1))

        return outs, loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs, encoder_states = self.encoder(x)
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(self.vocab.bos_idx)
        
        decoder_hidden, decoder_memory = encoder_states
        outputs = []
        for _ in range(self.MAX_LENGTH):
            decoder_output, (decoder_hidden, decoder_memory) = self.decoder.forward_step(decoder_input, (decoder_hidden, decoder_memory))
            decoder_input = decoder_output.argmax(dim=-1)
            outputs.append(decoder_input)

            if (decoder_input == self.vocab.eos_idx).all():
                break

        outputs = torch.cat(outputs, dim=1)
        
        return outputs

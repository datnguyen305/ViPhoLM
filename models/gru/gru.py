import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

# Define the GRU Encoder
class Encoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.gru = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            bidirectional=config.bidirectional,
            num_layers=config.layer_dim,
            batch_first=True,
            dropout=config.dropout
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

# Define the GRU Decoder

class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.hidden_size = config.hidden_size

        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.gru = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            num_layers=config.layer_dim,
            batch_first=True,
            dropout=config.dropout
        )
        self.out = nn.Linear(config.hidden_size, vocab.vocab_size)

    def forward(self, encoder_outputs: torch.Tensor, hidden: torch.Tensor, target_tensor: torch.Tensor, use_teacher_forcing=True):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size, 1), self.vocab.bos_idx, dtype=torch.long, device=encoder_outputs.device)
        
        decoder_outputs = []
        target_len = target_tensor.size(1)
        for t in range(target_len):
            decoder_output, hidden = self.forward_step(decoder_input, hidden)
            decoder_outputs.append(decoder_output)

            if use_teacher_forcing:
                decoder_input = target_tensor[:, t].unsqueeze(1)  # Teacher forcing
            else:
                decoder_input = decoder_output.argmax(dim=-1).unsqueeze(1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        
        return decoder_outputs, hidden

    def forward_step(self, input, hidden):
        embedded = self.embedding(input)
        embedded = F.relu(embedded)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output)

        return output, hidden


@META_ARCHITECTURE.register()
class GRU_Model(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2 
        self.d_model = config.d_model

        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)

        self.loss = nn.CrossEntropyLoss()

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
            
            if decoder_input.item() == self.vocab.eos_idx:
                break

        outputs = torch.cat(outputs, dim=1)
        
        return outputs


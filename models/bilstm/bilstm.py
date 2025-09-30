import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class Encoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab.vocab_size,
            config.hidden_size,
            device=config.device
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.lstm = nn.LSTM(
            config.hidden_size*2,  # because bidirectional
            config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=True,
            dropout=config.dropout,
            batch_first=True,
            device=config.device
        )
    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        outputs, (hidden,cell) = self.lstm(embedded)
        return outputs, (hidden,cell)
    
class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab.vocab_size,
            config.hidden_size,
            device=config.device
        )

        self.lstm = nn.LSTM(
            config.hidden_size*2,  # because bidirectional
            config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True,
            device=config.device
        )

        self.linear = nn.Linear(
            config.hidden_size,
            vocab.vocab_size,
            device=config.device
        )

    def forward(self, encoder_outputs, encoder_states, target_tensor):
        # Teacher forcing: feed the target as the next input
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size,1,dtype=torch.long,device=encoder_outputs.device).fill_(self.vocab.bos_idx)
        decoder_outputs = []
        target_len = target_tensor.shape[-1]
        for i in range(target_len):
            decoder_output, (decoder_hidden, decoder_memory) = self.forward_step(decoder_input, (encoder_states[0], encoder_states[1]))
            decoder_outputs.append(decoder_output)
            decoder_input = target_tensor[:, i].unsqueeze(1)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, (decoder_hidden, decoder_memory)

    def forward_step(self, input, states):
        embedded = self.embedding(input)
        output = F.relu(embedded)
        output, (hidden, memory) = self.lstm(output, states)
        output = self.linear(output)
        return output, (hidden, memory)
    
@META_ARCHITECTURE.register()
class BiLSTMModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2 

        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        encoder_outs, hidden_states = self.encoder(x)
        decoder_outs,_ = self.decoder(encoder_outs, hidden_states, labels)

        loss = self.loss(decoder_outs.reshape(-1, self.vocab.vocab_size), labels.reshape(-1))
        return decoder_outs, loss
    
    def predict(self, x): 
        encoder_outputs, encoder_states = self.encoder(x)
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(self.vocab.bos_idx)
        outputs = []
        for _ in range(self.MAX_LENGTH):
            decoder_output, (decoder_hidden, decoder_memory) = self.decoder.forward_step(decoder_input, (encoder_states[0], encoder_states[1]))
            decoder_input = decoder_output.argmax(dim=-1)  # Always pick best
            outputs.append(decoder_input)

            if decoder_input.item() == self.vocab.eos_idx:
                break
        
        outputs = torch.cat(outputs, dim=1)
        return outputs
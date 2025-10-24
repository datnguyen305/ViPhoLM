import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class Encoder(nn.Module):
    def __init__(self, config, vocab):
        super(Encoder, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        
        # BiGRU layer
        self.bigru = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            num_layers=config.layer_dim, 
            bidirectional= config.bidirectional,  # Bi-directional GRU
            batch_first=True, 
            dropout=config.dropout
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = config.hidden_size

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))  # [batch_size, seq_len, hidden_size]
        
        output, hidden = self.bigru(embedded)    # [batch_size, seq_len, hidden_size * 2], [num_layers * 2, batch_size, hidden_size]
        
        if self.bigru.bidirectional:
            hidden = self._combine_bidirectional_hidden(hidden) # [num_layers, batch_size, hidden_size]
        
        return output, hidden
    
    def _combine_bidirectional_hidden(self, hidden):
        num_layers = hidden.size(0) // 2  
        hidden = hidden.view(num_layers, 2, hidden.size(1), hidden.size(2))
        combined_hidden = hidden.sum(dim=1)
        return combined_hidden  # [num_layers, batch_size, hidden_size]


class Decoder(nn.Module):
    def __init__(self, config, vocab):
        super(Decoder, self).__init__()

        self.vocab = vocab
        
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        
        self.bigru = nn.GRU(
            config.hidden_size,
            config.hidden_size, 
            num_layers=config.layer_dim, 
            bidirectional=config.bidirectional,  # Bi-directional GRU
            batch_first=True, 
            dropout=config.dropout
        )
        
        multiplier = 2 if config.bidirectional else 1
        self.out = nn.Linear(config.hidden_size * multiplier, vocab.vocab_size)  # *2 due to bidirectional

    def forward(self, encoder_outputs: torch.Tensor, encoder_hidden: torch.Tensor, target_tensor: torch.Tensor):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(self.vocab.bos_idx)
        
        # dieu chinh hidden_state cho bigru
        decoder_hidden = torch.cat([encoder_hidden, encoder_hidden], dim=0)

        decoder_outputs = []
        target_len = target_tensor.shape[1]
        
        for i in range(target_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            
            # Teacher forcing
            decoder_input = target_tensor[:, i].unsqueeze(1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim = -1)
        
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        output = self.embedding(input)  # [batch_size, 1, hidden_size]
        output = F.relu(output)
        
        output, hidden = self.bigru(output, hidden)

        output = self.out(output)   # [batch_size, vocab_size]
        
        return output, hidden

@META_ARCHITECTURE.register()
class BiGRU_Model(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(BiGRU_Model, self).__init__()

        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2 # +2 for bos and eos tokens
        self.d_model = config.d_model

        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx)

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        encoder_outputs, encoder_hidden = self.encoder(x)
        decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, labels)
        
        loss = self.loss_fn(decoder_outputs.reshape(-1, self.vocab.vocab_size), labels.reshape(-1))
        return decoder_outputs, loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs, encoder_hidden = self.encoder(x)
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(self.vocab.bos_idx)
        
        decoder_hidden = encoder_hidden
        outputs = []
        
        for _ in range(self.MAX_LENGTH):
            decoder_output, decoder_hidden = self.decoder.forward_step(decoder_input, decoder_hidden)
            decoder_input = decoder_output.argmax(dim=-1).unsqueeze(1)  # [batch_size, 1]
            outputs.append(decoder_input)
            
            if (decoder_input == self.vocab.eos_idx).all():
                break
        
        outputs = torch.cat(outputs, dim=1) # [batch_size, seq_len]
        return outputs

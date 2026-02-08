import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, n_hidden_enc, n_hidden_dec):
        super().__init__()
        
        self.n_hidden_enc = n_hidden_enc
        self.n_hidden_dec = n_hidden_dec
        
        # W transforms concatenated encoder outputs and decoder hidden state
        self.W = nn.Linear(n_hidden_enc + n_hidden_dec, n_hidden_dec, bias=False)
        # V is the attention vector
        self.V = nn.Parameter(torch.rand(n_hidden_dec))
    
    def forward(self, hidden_dec, encoder_outputs):
        '''
        PARAMS:
            hidden_dec: [batch, n_hidden_dec] - current decoder hidden state
            encoder_outputs: [batch, seq_len, n_hidden_enc] - all encoder outputs
        
        RETURN:
            attention_weights: [batch, seq_len]
            context: [batch, n_hidden_enc]
        '''
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat decoder hidden state for each encoder output
        hidden_dec_repeated = hidden_dec.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, n_hidden_dec]
        
        # Concatenate and transform
        combined = torch.cat((hidden_dec_repeated, encoder_outputs), dim=2)  # [batch, seq_len, n_hidden_enc + n_hidden_dec]
        energy = torch.tanh(self.W(combined))  # [batch, seq_len, n_hidden_dec]
        
        # Calculate attention scores
        energy = energy.permute(0, 2, 1)  # [batch, n_hidden_dec, seq_len]
        V = self.V.repeat(batch_size, 1).unsqueeze(1)  # [batch, 1, n_hidden_dec]
        attention_scores = torch.bmm(V, energy).squeeze(1)  # [batch, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len]
        
        # Calculate context vector (weighted sum of encoder outputs)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, n_hidden_enc]
        context = context.squeeze(1)  # [batch, n_hidden_enc]
        
        return attention_weights, context
    

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
            config.hidden_size + config.hidden_size, # embedding + context vector
            lstm_hidden_size,
            bidirectional=config.bidirectional,
            num_layers=config.layer_dim,
            batch_first=True,
            dropout=config.dropout
        )
        
        self.attention = Attention(config.hidden_size, config.hidden_size)

        # LSTM out + context vector + embedding
        self.out = nn.Linear(config.hidden_size * 2 + config.hidden_size, vocab.vocab_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, encoder_outputs: torch.Tensor, encoder_states: torch.Tensor, target_tensor: torch.Tensor):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(self.vocab.bos_idx)
        
        decoder_hidden, decoder_memory = encoder_states
        decoder_outputs = []
        attention_weights_list = []
        target_len = target_tensor.shape[-1]

        for i in range(target_len):
            decoder_output, (decoder_hidden, decoder_memory), attn_weights = self.forward_step(decoder_input, (decoder_hidden, decoder_memory), encoder_outputs)
            decoder_outputs.append(decoder_output)
            attention_weights_list.append(attn_weights)
            # teacher forcing 
            decoder_input = target_tensor[:, i].unsqueeze(1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attention_weights = torch.stack(attention_weights_list, dim=1)  # [batch_size, target_len, src_len]
        return decoder_outputs, (decoder_hidden, decoder_memory), attention_weights

    def forward_step(self, input, states, encoder_outputs):
        '''
        input: [batch, 1] - input token
        states: tuple of (hidden, memory) for LSTM
        encoder_outputs: [batch, seq_len, hidden_size] - all encoder outputs
        '''
        # Get current hidden state for attention
        hidden, memory = states
        # Use last layer's hidden state for attention
        current_hidden = hidden[-1] if hidden.dim() == 3 else hidden  # [batch, hidden_size]
        
        # Calculate attention
        attn_weights, context = self.attention(current_hidden, encoder_outputs)
        
        # Embed input
        embedded = self.dropout(self.embedding(input))  # [batch, 1, hidden_size]
        
        # Concatenate embedding with context
        context_expanded = context.unsqueeze(1)  # [batch, 1, hidden_size]
        lstm_input = torch.cat((embedded, context_expanded), dim=2)  # [batch, 1, hidden_size*2]
        
        # LSTM forward
        lstm_output, (hidden, memory) = self.lstm(lstm_input, states)
        
        # Prepare final output by combining lstm output, context, and embedding
        combined = torch.cat((
            lstm_output.squeeze(1),  # [batch, hidden_size]
            context,  # [batch, hidden_size]
            embedded.squeeze(1)  # [batch, hidden_size]
        ), dim=1)  # [batch, hidden_size*3]
        
        output = self.out(combined).unsqueeze(1)  # [batch, 1, vocab_size]
        
        return output, (hidden, memory), attn_weights

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

        outs, _, attention_weights = self.decoder(encoder_outs, hidden_states, labels)

        loss = self.loss(outs.reshape(-1, self.vocab.vocab_size), labels.reshape(-1))

        return outs, loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs, encoder_states = self.encoder(x)
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(self.vocab.bos_idx)
        
        decoder_hidden, decoder_memory = encoder_states
        outputs = []
        for _ in range(self.MAX_LENGTH):
            decoder_output, (decoder_hidden, decoder_memory), _ = self.decoder.forward_step(decoder_input, (decoder_hidden, decoder_memory), encoder_outputs)
            decoder_input = decoder_output.argmax(dim=-1)
            outputs.append(decoder_input)

            if decoder_input == self.vocab.eos_idx:
                break

        outputs = torch.cat(outputs, dim=1)
        
        return outputs

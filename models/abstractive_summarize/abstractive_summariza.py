import torch 
import torch.nn as nn
import torch.nn.functional as F
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

"""
Encoder:
    self.embedding
    config: 
        vocab_size,
        embedding_dim,
        hidden-dim,
    self.lstm
    config:
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
        bidirectional,
        device

"""

class HierachyEncoder(nn.Module):
    def __init__(self, vocab: Vocab, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            vocab.vocab_size,
            config.embedding_dim
        )
        self.lstm = nn.LSTM(
            config.embedding_dim, 
            config.hidden_dim,
            num_layers=config.num_layers,
            dropout = config.dropout, 
            batch_first=True,
            bidirectional=config.bidirectional,
            device=config.device
        )
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell) # hidden, cell: (num_layers * num_directions, B, hidden_dim)

def create_mask(encoder_outputs, vocab: Vocab):
    mask = (encoder_outputs.sum(dim=-1) != vocab.pad_token).to(encoder_outputs.device)
    return mask
    
class BahdanauAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W1 = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.W2 = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.V = nn.Linear(config.hidden_dim, 1)
        self.W3 = nn.Linear(config.hidden_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden, mask):
        # encoder_outputs: (B, S, 2*hidden_size)
        # decoder_hidden: (B, 2*hidden_size))

        # Additive attention
        scores = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden.unsqueeze(1)))).squeeze(-1)
        # (B, S, 2*hidden_size) + (B, 1, hidden_size) -> (B, S, hidden_size) -> (B, S)
        scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        alphas = F.softmax(scores, dim=-1)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas.unsqueeze(1), encoder_outputs)

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        return context, alphas
    
class Decoder(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.config = config 
        self.vocab = vocab
        self.embedding = nn.Embedding(
            vocab.tgt_vocab_size,
            config.embedding_dim,
            padding_idx=vocab.pad_idx
        )
        self.lstm = nn.LSTM(
            config.embedding_dim + (config.hidden_dim *2),
            config.hidden_dim * 2, 
            num_layers=config.num_layers,
            dropout = config.dropout,
            batch_first=True,
            bidirectional=False,
            device=config.device
        )
        self.dropout = nn.Dropout(config.dropout)   
        self.attention = BahdanauAttention(config)
        self.fc_out = nn.Linear(config.hidden_dim*2, vocab.tgt_vocab_size)
    def forward(self, input, encoder_outputs, states, target):
        batch_size = target.size(0)
        target_len = target.size(1)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.vocab.bos_idx).to(self.config.device)
        decoder_outputs = []
        attn_weights = []
        encoder_output_mask = create_mask(encoder_outputs, self.vocab.pad_idx)
        decoder_states = states
        for i in range(target_len): 
            decoder_output, decoder_states, attn_weight = self.forward_step(
                decoder_input, 
                decoder_states, 
                encoder_outputs, 
                encoder_output_mask
                )
            decoder_outputs.append(decoder_output)
            attn_weights.append(attn_weight)
            # Here comes the teacher forcing
            decoder_input = target[:, i].unsqueeze(1)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_states
            

    def forward_step(self, input, state, encoder_outputs, mask):
        embedded = self.dropout(self.embedding(input)) # (B, 1, embedding_dim)
        hidden, _ = state 
        # hidden (num_layers, B, hidden_dim*2)
        # encoder_outputs (B, S, hidden_dim*2)
        # hidden[-1] bây giờ có shape (B, 2H)
        context, attn_weights = self.attention(encoder_outputs, hidden[-1], mask)
        # context (B, 1, hidden_dim)
        # embedded (B, 1, embedding_dim)
        input_lstm = torch.cat((embedded, context), dim=2)
        # input_lstm (B, 1, embedding_dim + hidden_dim)

        output, state = self.lstm(input_lstm, state)
        output = output.squeeze(1)  # (B, hidden_dim*2)
        logit = self.fc_out(output)
        return logit, state, attn_weights  
    
class BahdanauLSTM(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.encoder = HierachyEncoder(vocab, config)
        self.decoder = Decoder(vocab, config)
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.vocab = vocab
        self.config = config
    def forward(self, src, tgt):
        encoder_outputs, encoder_states = self.encoder(src)
        hidden_reshaped, cell_reshaped = self._reshape_encoder_states(encoder_states[0], encoder_states[1])
        encoder_states = (hidden_reshaped, cell_reshaped)
        decoder_outputs, _ = self.decoder(src, encoder_outputs, encoder_states, tgt)
        loss = self.loss(
            decoder_outputs.reshape(-1, self.vocab.tgt_vocab_size),
            tgt.reshape(-1)
        )
        return loss
    def _reshape_encoder_states(self, hidden, cell):
        if self.config.bidirectional:
            hidden = hidden.view(self.config.num_layers, 2, hidden.shape[1], hidden.shape[2])
            cell = cell.view(self.config.num_layers, 2, cell.shape[1], cell.shape[2])
            hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
            cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
        return hidden, cell
    def predict(self, src, tgt):
        encoder_outputs, encoder_states = self.encoder(src)
        hidden_reshaped, cell_reshaped = self._reshape_encoder_states(encoder_states[0], encoder_states[1])
        encoder_states = (hidden_reshaped, cell_reshaped)
        batch_size = src.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.vocab.bos_idx).to(self.config.device)
        decoder_states = encoder_states
        predictions = []
        max_len = tgt.size(1)
        encoder_output_mask = create_mask(encoder_outputs, self.vocab.pad_idx)
        for _ in range(max_len):
            decoder_output, decoder_states, _ = self.decoder.forward_step(
                decoder_input,
                decoder_states,
                encoder_outputs,
                encoder_output_mask
            )
            predicted_tokens = decoder_output.argmax(dim=-1).unsqueeze(1)
            predictions.append(predicted_tokens)
            decoder_input = predicted_tokens
        predictions = torch.cat(predictions, dim=1)
        return predictions
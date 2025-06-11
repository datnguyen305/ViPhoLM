import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.vocab = vocab
        self.hidden_size = config.hidden_size
        
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx=vocab.pad_idx)
        self.lstm = nn.LSTM(config.hidden_size + 2*config.hidden_size, config.hidden_size,
                            num_layers=config.layer_dim, batch_first=True, dropout=config.dropout)
        
        self.attention = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.attention_v = nn.Linear(config.hidden_size, 1, bias=False)
        
        self.pointer = nn.Linear(3 * config.hidden_size, 1)
        self.generator = nn.Linear(config.hidden_size, vocab.vocab_size)
        
        self.coverage = nn.Linear(1, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward_step(self, input, states, encoder_outputs, copy_probs, coverage, src_extended=None):
        decoder_hidden = states[0][-1].unsqueeze(1)
        attn_features = torch.cat([encoder_outputs, decoder_hidden.expand(-1, encoder_outputs.size(1), -1),
                                  self.coverage(coverage.unsqueeze(-1))], dim=-1)
        
        attn_energy = torch.tanh(self.attention(attn_features))
        attn_weights = F.softmax(self.attention_v(attn_energy).squeeze(-1), dim=1)
        coverage = coverage + attn_weights
        
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        embedded = self.dropout(self.embedding(input))
        lstm_input = torch.cat([embedded.squeeze(1), context], dim=-1)
        
        output, states = self.lstm(lstm_input.unsqueeze(1), states)
        p_gen = torch.sigmoid(self.pointer(torch.cat([context, output.squeeze(1)], dim=-1)))
        vocab_dist = F.softmax(self.generator(output.squeeze(1)), dim=-1)

        final_dist = torch.zeros(input.size(0), self.vocab.vocab_size).to(input.device)
        final_dist[:, :self.vocab.vocab_size] = p_gen * vocab_dist
        return final_dist, states, attn_weights, coverage

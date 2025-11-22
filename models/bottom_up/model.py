import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


class ContentSelector(nn.Module):
    """Content Selector used in Bottom-Up Abstractive Summarization"""
    def __init__(self, hidden_size):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, encoder_outputs):
        return torch.sigmoid(self.scorer(encoder_outputs))

class Encoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.bidirectional = config.bidirectional
        self.hidden_size = config.hidden_size

        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx=vocab.pad_idx)
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            bidirectional=config.bidirectional,
            num_layers=config.layer_dim,
            batch_first=True,
            dropout=config.dropout
        )
        self.content_selector = ContentSelector(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input, input_lengths):
        embedded = self.dropout(self.embedding(input))

        # lengths trên cpu, kiểu long
        input_lengths = input_lengths.cpu().long()

        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)

        outputs, (h, c) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        copy_probs = self.content_selector(outputs)
        copy_probs = copy_probs.squeeze(-1)

        return outputs, (h, c), copy_probs
    

class Decoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.vocab = vocab
        self.config = config 
        self.bidirectional = config.bidirectional
        self.hidden_size = config.hidden_size

        self.epsilon = 1e-8

        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx=vocab.pad_idx)
        # Decoder LSTM
        self.lstm = nn.LSTM(
            config.hidden_size + 2 * self.hidden_size,  # Input feeding
            config.hidden_size,
            num_layers=config.layer_dim,
            batch_first=True,
            dropout=config.dropout
        )

        # Always create reduction layers - they'll be used for bidirectional
        self.reduce_h = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.reduce_c = nn.Linear(2 * config.hidden_size, config.hidden_size)
        
        # Attention mechanism - adjust dimensions based on debug output
        # Encoder (1024) + decoder hidden (512) + coverage (512)
        self.attention = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.attention_v = nn.Linear(config.hidden_size, 1, bias=False)
        
        # Pointer-generator
        self.pointer = nn.Linear(3 * config.hidden_size, 1)
        self.generator = nn.Linear(config.hidden_size, vocab.vocab_size)
        
        self.coverage = nn.Linear(1, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward_step(self, input, states, encoder_outputs, copy_probs, coverage, src_extended=None):
        decoder_hidden = states[0][-1].unsqueeze(1)  # Shape: [batch_size, 1, hidden_size]
        expanded_hidden = decoder_hidden.expand(-1, encoder_outputs.size(1), -1)
        coverage_feature = self.coverage(coverage.unsqueeze(-1))

        # Build attention features with correct dimensions
        attn_features = torch.cat([
            encoder_outputs,
            expanded_hidden,
            coverage_feature
        ], dim=-1)
        
        attn_energy = torch.tanh(self.attention(attn_features))
        attn_weights = F.softmax(self.attention_v(attn_energy).squeeze(-1), dim=1)
        
        # Apply mask bottom-up
        if copy_probs is not None:
            mask = (copy_probs > self.epsilon).float()
            attn_weights = attn_weights * mask
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-10)
            
        # Update coverage
        coverage = coverage + attn_weights
        
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        embedded = self.dropout(self.embedding(input))
        lstm_input = torch.cat([embedded.squeeze(1), context], dim=-1)
        
        output, states = self.lstm(lstm_input.unsqueeze(1), states)
        
        p_gen = torch.sigmoid(self.pointer(torch.cat([context, output.squeeze(1)], dim=-1)))
        vocab_dist = F.softmax(self.generator(output.squeeze(1)), dim=-1)
        
        if src_extended is not None:
            extended_vocab_size = self.vocab.vocab_size + int(src_extended.max().item()) + 1
            final_dist = torch.zeros(input.size(0), extended_vocab_size).to(input.device)
        else:
            final_dist = torch.zeros(input.size(0), self.vocab.vocab_size).to(input.device)
        
        final_dist[:, :self.vocab.vocab_size] = p_gen * vocab_dist
        if src_extended is not None:
            final_dist.scatter_add_(1, src_extended, (1-p_gen) * attn_weights)
        
        return final_dist, states, attn_weights, coverage
    
    def forward(self, hidden):
        # We don't need to transform if we're doing it in the forward method
        return hidden
    
@META_ARCHITECTURE.register()
class BottomUpSummarizer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.config = config
        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)
        self.d_model = config.d_model
        self.device = config.device
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.loss_selector = nn.BCELoss(reduction='none')
        self.lambda_coverage = config.lambda_coverage if hasattr(config, 'lambda_coverage') else 1.0
        self.hidden_size = config.hidden_size

    def compute_loss(self, outputs, targets, copy_probs, attentions, x, src_oov=None):
        # Flatten outputs and targets for loss calculation
        outputs_flat = outputs.reshape(-1, outputs.size(-1))  # Changed from view to reshape
        targets_flat = targets.reshape(-1)  # Changed from view to reshape
        
        # Calculate cross-entropy loss
        loss_ce = self.loss_ce(outputs_flat, targets_flat)
        
        # Calculate coverage loss if needed
        if self.lambda_coverage > 0:
            # Create a binary mask for content selection
            mask = (x != self.vocab.pad_idx).float()
            loss_selector = self.loss_selector(copy_probs, mask)
            loss_selector = (loss_selector * mask).sum() / mask.sum()
            return loss_ce + self.lambda_coverage * loss_selector
        else:
            return loss_ce

    def forward(self, x: torch.Tensor, labels: torch.Tensor, src_oov=None):
        if src_oov is None:
            src_oov = x

        src_lengths = (x != self.vocab.pad_idx).sum(dim=1)
        
        # Encoder step
        encoder_outputs, encoder_states, copy_probs = self.encoder(x, src_lengths)
        
        # Prepare decoder input
        batch_size = x.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=x.device).fill_(self.vocab.bos_idx)
        
        # Handle bidirectional encoder states
        h, c = encoder_states
        num_layers = self.encoder.lstm.num_layers
        
        if self.encoder.bidirectional:
            # Reshape and process bidirectional hidden states based on debug output
            h_view = h.view(num_layers, 2, batch_size, -1)
            c_view = c.view(num_layers, 2, batch_size, -1)
            
            h_reshaped = []
            c_reshaped = []
            
            for layer in range(num_layers):
                # Concatenate forward and backward directions for each layer
                h_concat = torch.cat([h_view[layer, 0], h_view[layer, 1]], dim=-1)
                c_concat = torch.cat([c_view[layer, 0], c_view[layer, 1]], dim=-1)
                
                # Apply dimension reduction to match decoder hidden size
                h_reduced = F.relu(self.decoder.reduce_h(h_concat))
                c_reduced = F.relu(self.decoder.reduce_c(c_concat))
                
                h_reshaped.append(h_reduced)
                c_reshaped.append(c_reduced)
            
            # Stack the layers back
            h = torch.stack(h_reshaped, dim=0)
            c = torch.stack(c_reshaped, dim=0)
        
        # Initialize coverage
        coverage = torch.zeros(batch_size, x.size(1), dtype=torch.float, device=x.device)
        
        # Get initial decoder states
        decoder_states = (h, c)
        
        outputs = []
        attentions = []
        for di in range(labels.size(1)-1):  # Skip last token
            final_dist, decoder_states, attn_weights, coverage = self.decoder.forward_step(
                decoder_input, decoder_states, encoder_outputs, 
                copy_probs, coverage, src_extended=src_oov)
            
            outputs.append(final_dist)
            attentions.append(attn_weights)
            
            # Teacher forcing
            decoder_input = labels[:, di+1].unsqueeze(1)
        
        outputs = torch.stack(outputs, dim=1)
        attentions = torch.stack(attentions, dim=1)
        
        # Compute loss
        loss = self.compute_loss(outputs, labels[:, 1:], copy_probs, attentions, x, src_oov)
        
        return outputs, loss
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor, src_oov=None, max_len=100) -> torch.Tensor:
        self.eval()

        if src_oov is None:
            src_oov = x

        src_lengths = (x != self.vocab.pad_idx).sum(dim=1)
        encoder_outputs, encoder_states, copy_probs = self.encoder(x, src_lengths)
        
        batch_size = x.size(0)
        decoder_input = torch.full((batch_size, 1), self.vocab.bos_idx, dtype=torch.long, device=x.device)

        # Handle bidirectional encoder states
        h, c = encoder_states
        num_layers = self.encoder.lstm.num_layers
        if self.encoder.bidirectional:
            h_view = h.view(num_layers, 2, batch_size, -1)
            c_view = c.view(num_layers, 2, batch_size, -1)

            h_reshaped, c_reshaped = [], []
            for layer in range(num_layers):
                h_concat = torch.cat([h_view[layer, 0], h_view[layer, 1]], dim=-1)
                c_concat = torch.cat([c_view[layer, 0], c_view[layer, 1]], dim=-1)

                h_reduced = F.relu(self.decoder.reduce_h(h_concat))
                c_reduced = F.relu(self.decoder.reduce_c(c_concat))

                h_reshaped.append(h_reduced)
                c_reshaped.append(c_reduced)

            h = torch.stack(h_reshaped, dim=0)
            c = torch.stack(c_reshaped, dim=0)

        decoder_states = (h, c)
        coverage = torch.zeros(batch_size, x.size(1), dtype=torch.float, device=x.device)

        predictions = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(max_len):
            final_dist, decoder_states, attn_weights, coverage = self.decoder.forward_step(
                decoder_input, decoder_states, encoder_outputs, 
                copy_probs, coverage, src_extended=src_oov)

            top1 = final_dist.argmax(dim=-1)  # [batch_size]
            predictions.append(top1.unsqueeze(1))
            decoder_input = top1.unsqueeze(1)

            finished = finished | (top1 == self.vocab.eos_idx)
            if finished.all():
                break

        return torch.cat(predictions, dim=1)  # [batch_size, seq_len]

# python3 main.py --config-file configs/bottom_up_VietNews.yaml



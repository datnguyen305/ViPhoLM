import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


class Encoder(nn.Module):
    def __init__(self, config, vocab):
        super(Encoder, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx=vocab.pad_idx)
        
        # BiGRU layer - Encoder có thể dùng bidirectional
        self.bigru = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            num_layers=config.layer_dim, 
            bidirectional=config.bidirectional,
            batch_first=True, 
            dropout=config.dropout if config.layer_dim > 1 else 0  # GRU chỉ apply dropout khi > 1 layer
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = config.hidden_size
        self.bidirectional = config.bidirectional

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))  # [batch_size, seq_len, hidden_size]
        
        output, hidden = self.bigru(embedded)  # output: [B, L, H*2], hidden: [num_layers*2, B, H]
        
        # Combine bidirectional hidden states
        if self.bidirectional:
            hidden = self._combine_bidirectional_hidden(hidden)  # [num_layers, B, H]
        
        return output, hidden
    
    def _combine_bidirectional_hidden(self, hidden):
        """
        Combine forward and backward hidden states
        Input: [num_layers * 2, batch_size, hidden_size]
        Output: [num_layers, batch_size, hidden_size]
        """
        num_layers = hidden.size(0) // 2  
        # Reshape: [num_layers, 2, batch_size, hidden_size]
        hidden = hidden.view(num_layers, 2, hidden.size(1), hidden.size(2))
        # Sum forward and backward: [num_layers, batch_size, hidden_size]
        combined_hidden = hidden.sum(dim=1)
        return combined_hidden


class Decoder(nn.Module):
    def __init__(self, config, vocab):
        super(Decoder, self).__init__()

        self.vocab = vocab
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx=vocab.pad_idx)
        
        self.gru = nn.GRU(
            config.hidden_size,
            config.hidden_size, 
            num_layers=config.layer_dim, 
            bidirectional=False,  
            batch_first=True, 
            dropout=config.dropout if config.layer_dim > 1 else 0
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.out = nn.Linear(config.hidden_size, vocab.vocab_size)

    def forward(self, encoder_outputs: torch.Tensor, encoder_hidden: torch.Tensor, target_tensor: torch.Tensor):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(self.vocab.bos_idx)
        
        decoder_hidden = encoder_hidden

        decoder_outputs = []
        target_len = target_tensor.shape[1]
        
        for i in range(target_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            
            # Teacher forcing: dùng target thật
            decoder_input = target_tensor[:, i].unsqueeze(1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        """
        Single decoding step
        """
        output = self.embedding(input)  # [batch_size, 1, hidden_size]
        output = self.dropout(output)
        
        output, hidden = self.gru(output, hidden)  # [batch_size, 1, hidden_size]

        output = self.out(output)  # [batch_size, 1, vocab_size]
        
        return output, hidden


@META_ARCHITECTURE.register()
class BiGRU_Model(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(BiGRU_Model, self).__init__()

        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2  # +2 for bos and eos tokens
        self.d_model = config.d_model

        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)

        label_smoothing = getattr(config, 'label_smoothing', 0.0)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.vocab.pad_idx,
            label_smoothing=label_smoothing
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        encoder_outputs, encoder_hidden = self.encoder(x)
        decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, labels)
        
        # Compute loss
        loss = self.loss_fn(
            decoder_outputs.reshape(-1, self.vocab.vocab_size), 
            labels.reshape(-1)
        )
        return decoder_outputs, loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference without teacher forcing
        """
        # Encode input
        encoder_outputs, encoder_hidden = self.encoder(x)
        batch_size = encoder_outputs.size(0)
        decoder_hidden = encoder_hidden
        
        # Start with <bos> token
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(self.vocab.bos_idx)

        outputs = []

        for _ in range(self.MAX_LENGTH):
            # Predict one step
            decoder_output, decoder_hidden = self.decoder.forward_step(decoder_input, decoder_hidden)

            # Get token with highest probability
            decoder_input = decoder_output.argmax(dim=-1)  # [batch, 1]
            outputs.append(decoder_input)

            # Early stopping if all sequences generated <eos>
            if (decoder_input == self.vocab.eos_idx).all():
                break

        # Concatenate all outputs: [batch, seq_len]
        outputs = torch.cat(outputs, dim=1)
        return outputs
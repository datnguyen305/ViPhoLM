import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.viword_vocab import ViWordVocab
from models.lstm_phoneme.layers.encoder import Encoder
from models.lstm_phoneme.layers.decoder import Decoder
from builders.model_builder import META_ARCHITECTURE


@META_ARCHITECTURE.register()
class LSTM_Model_Phoneme(nn.Module):
    def __init__(self, config, vocab: ViWordVocab):
        super().__init__()

        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2 # + 2 for bos and eos tokens
        self.d_model = config.d_model
        self.num_features = 3
        self.config = config

        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)
        
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        encoder_outs, hidden_states = self.encoder(x)
        # encoder_outs : (batch_size, seq_len, hidden_size * 3)
    
        outs, _ = self.decoder(encoder_outs, hidden_states, labels)
        # decoder_outputs: (batch_size, target_len, 3, vocab_size)
        B, L, _, vocab_size = outs.shape
        outs = outs.reshape(B, 3*L, -1) # (B, 3*L, vocab_size)
        loss = self.loss(outs.reshape(-1, vocab_size), labels)
    
        return outs, loss
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs, encoder_states = self.encoder(x)
        batch_size = encoder_outputs.size(0)

        # Initiate decoder's input [<BOS>, <BOS>, <BOS>]
        decoder_input = torch.empty(batch_size, 1, self.num_features, dtype=torch.long, device=self.config.device)
        decoder_input.fill_(self.vocab.bos_idx)
        # decoder_input: (batch_size, 1, 3)

        decoder_hidden, decoder_memory = encoder_states
        # decoder_hidden: (num_layers, batch_size, hidden_size * 3)
        # decoder_memory: (num_layers, batch_size, hidden_size * 3)

        outputs = []
        target_len = self.vocab.max_sentence_length

        for _ in range(target_len):
            decoder_output, (decoder_hidden, decoder_memory) = self.decoder.forward_step(decoder_input, (decoder_hidden, decoder_memory))
            # decoder_output: (batch_size, 1, 3, vocab_size)

            decoder_input = decoder_output.argmax(dim=-1)
            # decoder_input: (batch_size, 1, 3)
            outputs.append(decoder_input)
            # outputs: (batch_size, 1, 3) * seq_length

            if decoder_input[:, :, 0] == self.vocab.eos_idx:
                break
        
        outputs = torch.cat(outputs, dim=1)
        # outputs: (batch_size, seq_len, 3)

        return outputs
    
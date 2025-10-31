import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class Encoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab.vocab_size, config.hidden_size, device=config.device
        )
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
            dropout=0.7,
            device=config.device
        )
        self.linear = nn.Linear(config.hidden_size*2, config.hidden_size, device=config.device)

    def forward(self, input):
        """
        Input: (batch_size, seq_len)
        Output: 
            output: (batch_size, seq_len, hidden_size * 2)
            h_n: (2*num_layers, hidden_size)
            c_n: (2*num_layers, hidden_size)
        """
        embedded = self.embedding(input) 
        encoder_output, (h_n, c_n) = self.lstm(embedded)
        encoder_output = self.linear(encoder_output)
        encoder_input = input
        # output: (batch_size, seq_len, hidden_size)
        # h_n: (2*num_layers, batch_size, hidden_size)
        # c_n: (2*num_layers, batch_size, hidden_size)
        return encoder_output, (h_n, c_n), encoder_input

class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab.vocab_size,
            config.hidden_size,
            device=config.device
        )
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.7,
            device=config.device,
            bidirectional=False
        )
        self.attn = BahdanauAttention(config)
        self.out = nn.Linear(config.hidden_size, vocab.vocab_size, device=config.device)
        self.linear1 = nn.Linear(config.hidden_size * 2, config.hidden_size, device=config.device)
        self.linear2 = nn.Linear(config.hidden_size, vocab.vocab_size, device=config.device) 
        self.p_gen_linear = nn.Linear(config.hidden_size * 3, 1, device=config.device)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input, states, target_tensor=None, encoder_outputs=None, encoder_input=None):
        """
        Input: 
            input: (batch_size, 1)
            states: (h_n, c_n)
                h_n: (num_layers, batch_size, hidden_size)
                c_n: (num_layers, batch_size, hidden_size)
        Output:
            output: (batch_size, target_len, vocab_size)
            h_n: (num_layers, batch_size, hidden_size)
            c_n: (num_layers, batch_size, hidden_size)
        """
        decoder_hidden, decoder_memory = states
        decoder_outputs = []
        decoder_input = input  # Sử dụng trực tiếp input parameter
        target_len = target_tensor.shape[-1]
        for i in range(target_len):
            decoder_output, (decoder_hidden, decoder_memory) = self.forward_step(
                decoder_input, 
                (decoder_hidden, decoder_memory),
                encoder_input,
                encoder_outputs,
                num_oov_in_batch=0
            )
            decoder_outputs.append(decoder_output)
            # Teacher forcing: Feed the target as the next input
            decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing

        decoder_outputs = torch.cat(decoder_outputs, dim=1) # (batch_size, target_len, vocab_size)
        return decoder_outputs, (decoder_hidden, decoder_memory)

    def forward_step(self, input, states, encoder_input=None, encoder_outputs=None, num_oov_in_batch=0):
        embedded = self.embedding(input) # (batch_size, 1, hidden_size)
        output, (h_n, c_n) = self.lstm(embedded, states)
        # output : (batch_size, 1, hidden_size)
        # h_n : (num_layers, batch_size, hidden_size)
        # c_n : (num_layers, batch_size, hidden_size)
        output = output.squeeze(1)  # (batch_size, hidden_size)

        if encoder_outputs is None:
            raise ValueError("encoder_outputs cannot be None")

        # attention
        context_vector, attention_weights = self.attn(h_n[-1], encoder_outputs) # h_n[-1]: (batch_size, hidden_size)
        # context_vector: (batch_size, hidden_size)
        # attention_weights: (batch_size, seq_len)

        #P_vocab
        concat_input = torch.cat((h_n[-1], context_vector), dim=-1)  # (batch_size, hidden_size*2)
        hidden = F.relu(self.linear1(concat_input))
        logits = self.linear2(hidden)
        Pvocab = F.softmax(logits, dim=-1) # Pvocab: (batch_size, vocab_size)
        
        #P_gen
        input = input.squeeze(1)  # (batch_size, hidden_size)
        p_gen = self.sigmoid(self.p_gen_linear(torch.cat((context_vector, h_n[-1], input), dim=-1))) # (batch_size, 1)
        
        B = input.size(0)
        extended_vocab_size = self.vocab.vocab_size + num_oov_in_batch
        final_dist = torch.zeros(B, extended_vocab_size, device=Pvocab.device)
        
        # Final distribution
        vocab_dist = p_gen * Pvocab
        attn_dist = (1 - p_gen) * attention_weights


        if encoder_input is not None:
            final_dist.scatter_add_(1, encoder_input, attn_dist)
            final_dist[:, :Pvocab.size(1)] += vocab_dist
        else:
            final_dist = vocab_dist

        return final_dist, (h_n, c_n)
class BahdanauAttention(nn.Module): # Attention Bahdanau-style
    def __init__(self, config):
        super().__init__()
        self.W_h = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_s = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_a = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, prev_decoder_hidden, encoder_outputs): 
        """
        Inputs:
            prev_decoder_hidden: (batch_size, hidden_size)
            encoder_outputs: (B, S, hidden_size)
        Outputs:
            context_vector: (B, hidden_size)
            attention_weights: (B, S)
        """
        Wi_hi = self.W_h(encoder_outputs)  # (B, S, hidden_size)
        Ws_prev_s = self.W_s(prev_decoder_hidden)  # (B, hidden_size)
        E_ti = self.v_a(torch.tanh(Wi_hi + Ws_prev_s.unsqueeze(1))).squeeze(-1)  # (B, S)
        A_ti = F.softmax(E_ti, dim=1) # (B, S)
        C_t = F.bmm(A_ti.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, hidden_size)
        return C_t, A_ti  # context_vector, attention_weights

class ClosedBookModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab_size = vocab.vocab_size
        
        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)

        self.d_model = config.d_model
        self.device = config.device 
        self.config = config
        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2 # + 2 for bos and eos tokens
        self.loss = nn.CrossEntropyLoss()


    def forward(self, src, labels):
        """
        Forward pass for training
        
        Args:
            src (torch.Tensor): Source sequences of shape (batch_size, src_len)
            labels (torch.Tensor): Target sequences of shape (batch_size, tgt_len)
            
        Returns:
            tuple:
                - decoder_outputs (torch.Tensor): Output logits of shape (batch_size, tgt_len, vocab_size)
                - loss (torch.Tensor): Scalar loss value
        """
        oov_words = []
        for b in range(src.size(0)):
            for idx in encoder_input[b]:
                if idx.item() >= self.vocab.vocab_size and idx.item() not in oov_words:
                    oov_words.append(idx.item())
        num_oov_in_batch = len(oov_words)
        # Encode source sequence
        encoder_outputs, hidden_states, encoder_input = self.encoder(src)

        oov_words = []
        for b in range(src.size(0)):
            for idx in encoder_input[b]:
                if idx.item() >= self.vocab.vocab_size and idx.item() not in oov_words:
                    oov_words.append(idx.item())

        num_oov_in_batch = len(oov_words)
        
        # Initialize decoder input
        batch_size = src.size(0)
        decoder_input = torch.empty(
            batch_size, 1,
            dtype=torch.long,
            device=src.device
        ).fill_(self.vocab.bos_idx)
        
        # Decode with teacher forcing
        decoder_outputs, _ = self.decoder(
            decoder_input,
            hidden_states,
            labels,
            encoder_outputs,
            encoder_input,
            num_oov_in_batch=num_oov_in_batch
        )

        # Calculate loss
        loss = self.loss(
            decoder_outputs.reshape(-1, self.vocab_size),
            labels.reshape(-1)
        )
        
        return decoder_outputs, loss

    def predict(self, src: torch.Tensor) -> torch.Tensor:
        encoder_outputs, encoder_states, encoder_input = self.encoder(src)
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(self.vocab.bos_idx)
        
        decoder_hidden, decoder_memory = encoder_states
        outputs = []
        for _ in range(self.MAX_LENGTH):
            decoder_output, (decoder_hidden, decoder_memory) = self.decoder.forward_step(decoder_input, (decoder_hidden, decoder_memory), encoder_input)
            outputs.append(decoder_output)
            top1 = decoder_output.argmax(1)
            decoder_input = top1.unsqueeze(1)
        
        outputs = torch.stack(outputs, dim=1)  # (batch_size, max_length, vocab_size)
        return outputs
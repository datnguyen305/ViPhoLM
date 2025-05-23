from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import random
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab

def init_lstm_wt(lstm, config):
    """Initializes LSTM weights with uniform distribution and sets forget bias to 1."""
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear, config):
    """Initializes Linear layer weights with normal distribution."""
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt, config):
    """Initializes weights with normal distribution."""
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt, config):
    """Initializes weights with uniform distribution."""
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
    """
    Encoder module for the sequence-to-sequence model.
    Uses a bidirectional LSTM to process input sequences.
    """
    def __init__(self, config, vocab: Vocab):
        super(Encoder, self).__init__()
        self.config = config
        # Lấy vocab_size từ đối tượng vocab được truyền vào
        self.embedding = nn.Embedding(vocab.vocab_size, self.config.emb_dim)
        init_wt_normal(self.embedding.weight, self.config)
        
        self.lstm = nn.LSTM(self.config.emb_dim, self.config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm, self.config)

        self.W_h = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim * 2, bias=False)

    def forward(self, input, seq_lens):
        """
        Forward pass for the encoder.
        Args:
            input (Tensor): Input sequence tensor (batch_size, seq_len).
            seq_lens (Tensor): Lengths of sequences in the batch (batch_size).
        Returns:
            encoder_outputs (Tensor): Output hidden states from the encoder (batch_size, seq_len, 2*hidden_dim).
            encoder_feature (Tensor): Transformed encoder outputs for attention (batch_size * seq_len, 2*hidden_dim).
            hidden (tuple): Final hidden and cell states of the encoder LSTM.
        """
        embedded = self.embedding(input)
        
        # Pack padded sequence for efficient LSTM processing
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(packed)

        # Unpack sequence to get padded outputs
        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()
        
        # Prepare encoder features for attention mechanism
        encoder_feature = encoder_outputs.view(-1, 2*self.config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden

class ReduceState(nn.Module):
    """
    Reduces the bidirectional encoder's final hidden and cell states
    to a single-directional state for initializing the decoder.
    """
    def __init__(self, config):
        super(ReduceState, self).__init__()
        self.config = config # Store config
        # Linear layers to transform hidden and cell states
        self.reduce_h = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)
        init_linear_wt(self.reduce_h, self.config)
        self.reduce_c = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)
        init_linear_wt(self.reduce_c, self.config)

    def forward(self, hidden):
        """
        Forward pass to reduce encoder hidden states.
        Args:
            hidden (tuple): Hidden and cell states from the encoder (2, batch_size, hidden_dim).
        Returns:
            tuple: Reduced hidden and cell states for the decoder (1, batch_size, hidden_dim).
        """
        h, c = hidden # h, c dim = 2 x b x hidden_dim (bidirectional)
        
        # Concatenate forward and backward states and apply linear transformation
        h_in = h.transpose(0, 1).contiguous().view(-1, self.config.hidden_dim * 2) # batch_size x (2*hidden_dim)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        
        c_in = c.transpose(0, 1).contiguous().view(-1, self.config.hidden_dim * 2) # batch_size x (2*hidden_dim)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        # Unsqueeze to add a layer dimension (1 for single-layer decoder)
        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

class Attention(nn.Module):
    """
    Attention mechanism for the decoder. Calculates attention distribution
    over encoder outputs and computes the context vector.
    Supports coverage mechanism if config.is_coverage is True.
    """
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config # Store config
        # Coverage mechanism linear layer (if enabled)
        if self.config.is_coverage:
            self.W_c = nn.Linear(1, self.config.hidden_dim * 2, bias=False)
        
        # Linear layers for attention calculation
        self.decode_proj = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim * 2)
        self.v = nn.Linear(self.config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        """
        Forward pass for the attention network.
        Args:
            s_t_hat (Tensor): Decoder hidden state for attention (batch_size, 2*hidden_dim).
            encoder_outputs (Tensor): Encoder outputs (batch_size, seq_len, 2*hidden_dim).
            encoder_feature (Tensor): Pre-processed encoder outputs for attention (batch_size * seq_len, 2*hidden_dim).
            enc_padding_mask (Tensor): Mask for padded encoder inputs (batch_size, seq_len).
            coverage (Tensor): Current coverage vector (batch_size, seq_len) if config.is_coverage is True.
        Returns:
            c_t (Tensor): Context vector (batch_size, 2*hidden_dim).
            attn_dist (Tensor): Attention distribution (batch_size, seq_len).
            coverage (Tensor): Updated coverage vector.
        """
        b, t_k, n = list(encoder_outputs.size())

        # Project decoder hidden state for attention calculation
        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        # Calculate attention features
        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        
        # Add coverage feature if coverage mechanism is enabled
        if self.config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        # Calculate attention scores
        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        # Apply softmax and padding mask to get attention distribution
        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor # Normalize after masking

        # Calculate context vector
        attn_dist_for_bmm = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist_for_bmm, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, self.config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k (for return)

        # Update coverage vector if coverage mechanism is enabled
        if self.config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Decoder(nn.Module):
    """
    Decoder module for the sequence-to-sequence model.
    Generates output sequence step by step using an LSTM, attention,
    and optionally a pointer-generator mechanism with coverage.
    """
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config # Store config
        self.attention_network = Attention(self.config)
        
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.emb_dim)
        init_wt_normal(self.embedding.weight, self.config)

        # Linear layer to combine context vector and embedded input
        self.x_context = nn.Linear(self.config.hidden_dim * 2 + self.config.emb_dim, self.config.emb_dim)

        self.lstm = nn.LSTM(self.config.emb_dim, self.config.hidden_dim, num_layers=2, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm, self.config)

        # Pointer-generator linear layer (if enabled)
        if self.config.pointer_gen:
            self.p_gen_linear = nn.Linear(self.config.hidden_dim * 4 + self.config.emb_dim, 1)

        # Output layers for vocabulary distribution
        self.out1 = nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim)
        self.out2 = nn.Linear(self.config.hidden_dim, self.config.vocab_size)
        init_linear_wt(self.out2, self.config)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):
        """
        Forward pass for the decoder at a single time step.
        Args:
            y_t_1 (Tensor): Previous output token (batch_size).
            s_t_1 (tuple): Previous hidden and cell states of the decoder LSTM.
            encoder_outputs (Tensor): Encoder outputs (batch_size, seq_len, 2*hidden_dim).
            encoder_feature (Tensor): Pre-processed encoder outputs for attention (batch_size * seq_len, 2*hidden_dim).
            enc_padding_mask (Tensor): Mask for padded encoder inputs (batch_size, seq_len).
            c_t_1 (Tensor): Previous context vector (batch_size, 2*hidden_dim).
            extra_zeros (Tensor): Zeros for extended vocabulary (batch_size, max_oov_len) or None.
            enc_batch_extend_vocab (Tensor): Extended vocabulary indices for attention (batch_size, seq_len).
            coverage (Tensor): Current coverage vector (batch_size, seq_len).
            step (int): Current decoding step.
        Returns:
            final_dist (Tensor): Final probability distribution over vocabulary (and extended vocab if applicable).
            s_t (tuple): Current hidden and cell states of the decoder LSTM.
            c_t (Tensor): Current context vector.
            attn_dist (Tensor): Current attention distribution.
            p_gen (Tensor): Probability of generating from vocabulary vs. copying (batch_size, 1).
            coverage (Tensor): Updated coverage vector.
        """

        # Initial attention calculation for step 0 when not training.
        # This is a common practice to get the initial context vector.
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, self.config.hidden_dim),
                                 c_decoder.view(-1, self.config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        # Embed the previous output token and combine with the context vector
        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))

        # Pass through the decoder LSTM
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        # Get current decoder hidden state for attention
        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, self.config.hidden_dim),
                             c_decoder.view(-1, self.config.hidden_dim)), 1)  # B x 2*hidden_dim

        # Calculate new context vector and attention distribution
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)

        # Update coverage for the next step.
        # This update happens after the attention calculation for the current step.
        if self.training or step > 0:
            coverage = coverage_next

        # Calculate p_gen (probability of generating vs. copying) if pointer-generator is enabled in config
        p_gen = None
        if self.config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        # Calculate the vocabulary distribution
        output = torch.cat((lstm_out.view(-1, self.config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim
        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        # Determine the final distribution based on the 'closed_book_mode' flag from config.
        if self.config.closed_book_mode: # Use self.config.closed_book_mode here
            # In "closed-book" mode, the model relies ONLY on its learned vocabulary distribution.
            # The pointer-generator mechanism (copying from source) is effectively disabled.
            final_dist = vocab_dist
            # For consistency in return values, ensure p_gen is a tensor of ones if needed.
            if p_gen is None: # This case means config.pointer_gen was False initially
                 # Create a dummy p_gen of 1s, matching batch size
                 p_gen = torch.ones_like(vocab_dist[:, 0:1])
            else: # This case means config.pointer_gen was True, but we are in closed_book_mode
                 p_gen = torch.ones_like(p_gen) # Force p_gen to 1
        else:
            # If not in "closed-book" mode, proceed with the standard pointer-generator logic
            # if 'config.pointer_gen' is enabled.
            if self.config.pointer_gen:
                # Ensure p_gen was calculated (it should be if config.pointer_gen is True)
                if p_gen is None:
                    # Fallback/Error: p_gen should have been computed if config.pointer_gen is True
                    p_gen_input = torch.cat((c_t, s_t_hat, x), 1)
                    p_gen = self.p_gen_linear(p_gen_input)
                    p_gen = F.sigmoid(p_gen)

                vocab_dist_ = p_gen * vocab_dist
                attn_dist_ = (1 - p_gen) * attn_dist

                # If extended vocabulary is used (for OOV words), concatenate zeros and scatter_add
                if extra_zeros is not None:
                    vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

                final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
            else:
                # If 'config.pointer_gen' is False (and not closed_book_mode), just use vocabulary distribution.
                final_dist = vocab_dist
                p_gen = None # Explicitly set p_gen to None if not used

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage
    
@META_ARCHITECTURE.register()
class Closedbook(nn.Module):
    # ...
    def __init__(self, config, vocab: Vocab, model_file_path=None, is_eval=False):
        super(Closedbook, self).__init__()
        self.config = config

        # Gán vocab_size từ đối tượng vocab vào config của mô hình
        # Điều này đảm bảo rằng tất cả các lớp con (như Decoder) đều có vocab_size chính xác
        self.config.vocab_size = vocab.vocab_size

        self.device = config.device # <--- This line defines self.device

        # Truyền đối tượng vocab vào Encoder
        encoder = Encoder(self.config, vocab)
        # Decoder vẫn chỉ cần config vì vocab_size đã được cập nhật trong config
        decoder = Decoder(self.config)
        reduce_state = ReduceState(self.config)

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight

        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        encoder = encoder.to(self.device) # <--- self.device is used here
        decoder = decoder.to(self.device)
        reduce_state = reduce_state.to(self.device)

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False) 
            self.reduce_state.load_state_dict(state['reduce_state_dict'])

    def forward(self, enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, dec_batch, target_batch):
        """
        Performs a full forward pass of the summarization model for a batch.
        This method orchestrates the encoding and decoding process.

        Args:
            enc_batch (Tensor): Input article batch (batch_size, max_enc_steps).
            enc_lens (Tensor): Lengths of articles in the batch (batch_size).
            enc_padding_mask (Tensor): Mask for padded encoder inputs (batch_size, max_enc_steps).
            enc_batch_extend_vocab (Tensor): Extended vocabulary indices for attention (batch_size, max_enc_steps).
            extra_zeros (Tensor): Zeros for extended vocabulary (batch_size, max_oov_len) or None.
            dec_batch (Tensor): Decoder input batch (batch_size, max_dec_steps).
            target_batch (Tensor): Target output batch (batch_size, max_dec_steps).

        Returns:
            Tuple:
                - final_dists (list of Tensors): List of final probability distributions for each decoder step.
                - attn_dists (list of Tensors): List of attention distributions for each decoder step.
                - p_gens (list of Tensors): List of p_gen values for each decoder step.
        """
        # Move input tensors to the correct device
        enc_batch = enc_batch.to(self.device)
        enc_lens = enc_lens.to(self.device)
        enc_padding_mask = enc_padding_mask.to(self.device)
        enc_batch_extend_vocab = enc_batch_extend_vocab.to(self.device)
        if extra_zeros is not None:
            extra_zeros = extra_zeros.to(self.device)
        dec_batch = dec_batch.to(self.device)
        target_batch = target_batch.to(self.device) # Not directly used in forward, but good practice for consistency

        # 1. Encoder Forward Pass
        encoder_outputs, encoder_feature, hidden = self.encoder(enc_batch, enc_lens)

        # 2. Reduce Encoder State to initialize Decoder
        s_t_1 = self.reduce_state(hidden) # (h, c) for decoder LSTM

        # Initialize context vector and coverage
        c_t_1 = torch.zeros((self.config.batch_size, 2 * self.config.hidden_dim)).to(self.device)
        
        if self.config.is_coverage:
            coverage_t_1 = torch.zeros(enc_batch.size()).to(self.device) # B x max_enc_steps
        else:
            coverage_t_1 = None

        # Lists to store outputs from each decoder step
        final_dists = []
        attn_dists = []
        p_gens = []

        # 3. Decoder Forward Pass (step by step)
        for step in range(self.config.max_dec_steps):
            # Get input for current decoder step
            # For training, use teacher forcing: previous input is from dec_batch
            y_t_1 = dec_batch[:, step] # (batch_size)

            final_dist, s_t_1, c_t_1, attn_dist, p_gen, coverage_t_1 = \
                self.decoder(y_t_1, s_t_1, encoder_outputs, encoder_feature,
                             enc_padding_mask, c_t_1, extra_zeros,
                             enc_batch_extend_vocab, coverage_t_1, step)
            
            final_dists.append(final_dist)
            attn_dists.append(attn_dist)
            if self.config.pointer_gen or self.config.closed_book_mode:
                p_gens.append(p_gen) # p_gen is always returned if pointer_gen or closed_book_mode is true

        # Stack the list of tensors into a single tensor for easier handling
        # final_dists: (max_dec_steps, batch_size, vocab_size_extended)
        # attn_dists: (max_dec_steps, batch_size, max_enc_steps)
        # p_gens: (max_dec_steps, batch_size, 1)
        final_dists = torch.stack(final_dists, dim=0)
        attn_dists = torch.stack(attn_dists, dim=0)
        p_gens = torch.stack(p_gens, dim=0) if (self.config.pointer_gen or self.config.closed_book_mode) else None

        return final_dists, attn_dists, p_gens


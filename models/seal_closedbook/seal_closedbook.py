from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab
from models.transformer_seal.layers.seal_attention import SEALAttention  # Thêm dòng này


def init_lstm_wt(lstm, config, vocab: Vocab):
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

def init_linear_wt(linear, config, vocab: Vocab):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt, config, vocab: Vocab):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt, config, vocab: Vocab):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.hidden_dim = config.hidden_dim

    #seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)
       
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()
        
        encoder_feature = encoder_outputs.view(-1, 2*self.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden

class ReduceState(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)
        self.hidden_dim = config.hidden_dim

    def forward(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, self.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, self.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

# Xóa class Attention cũ

class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(Decoder, self).__init__()
        # attention_network dùng SEALAttention
        self.attention_network = SEALAttention(
            d_model=config.hidden_dim * 2,  # hoặc config.d_model nếu đúng
            n_head=getattr(config, 'n_head', 8),
            segment_size=getattr(config, 'segment_size', 512)
        )
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)
        self.pointer_gen = config.pointer_gen

        if self.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)
        self.hidden_dim = config.hidden_dim

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim),
                                 c_decoder.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
            # Sử dụng SEALAttention
            c_t = self.attention_network(encoder_outputs)
            coverage = coverage  # giữ nguyên

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim),
                             c_decoder.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
        # Sử dụng SEALAttention
        c_t = self.attention_network(encoder_outputs)

        if self.training or step > 0:
            coverage = coverage

        p_gen = None
        if self.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim
        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if self.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * c_t  # c_t ở đây là attention distribution
            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, c_t, p_gen, coverage

class ClosedbookDecoder(nn.Module):
    """
    Closedbook Decoder - sinh summary mà không cần source text
    Dựa trên ý tưởng "Closed-Book Training to Improve Summarization Encoder Memory"
    """
    def __init__(self, config, vocab: Vocab):
        super(ClosedbookDecoder, self).__init__()
        self.vocab = vocab
        self.hidden_dim = config.hidden_dim
        self.emb_dim = config.emb_dim
        self.embedding = nn.Embedding(vocab.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        self.internal_memory = nn.Parameter(torch.randn(1, config.max_src_len, config.hidden_dim * 2))
        self.memory_projection = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        init_linear_wt(self.memory_projection)
        # Attention network cho internal memory dùng SEALAttention
        self.memory_attention = SEALAttention(
            d_model=config.hidden_dim * 2,
            n_head=getattr(config, 'n_head', 8),
            segment_size=getattr(config, 'segment_size', 512)
        )
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, 
                           batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, vocab.vocab_size)
        init_linear_wt(self.out2)
        self.is_coverage = config.is_coverage
        if self.is_coverage:
            self.coverage_layer = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.pointer_gen = config.pointer_gen
        if self.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

    def forward(self, y_t_1, s_t_1, c_t_1, extra_zeros=None, 
                enc_batch_extend_vocab=None, coverage=None, step=0):
        batch_size = y_t_1.size(0)
        memory_len = self.internal_memory.size(1)
        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)
        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim),
                             c_decoder.view(-1, self.hidden_dim)), 1)
        encoder_outputs = self.internal_memory.expand(batch_size, -1, -1)
        encoder_feature = self.memory_projection(encoder_outputs.view(-1, self.hidden_dim * 2))
        encoder_feature = encoder_feature.view(batch_size, memory_len, self.hidden_dim * 2)
        enc_padding_mask = torch.ones(batch_size, memory_len, device=y_t_1.device)
        # Sử dụng SEALAttention
        c_t = self.memory_attention(encoder_outputs)
        if self.training or step > 0:
            coverage = coverage
        p_gen = None
        if self.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)
        output = torch.cat((lstm_out.view(-1, self.hidden_dim), c_t), 1)
        output = self.out1(output)
        output = self.out2(output)
        vocab_dist = F.softmax(output, dim=1)
        if self.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * c_t
            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist
        return final_dist, s_t, c_t, c_t, p_gen, coverage

    def x_context(self, input):
        """Context projection layer"""
        return input  # Simplified version, có thể mở rộng

@META_ARCHITECTURE.register()
class ClosedbookSeal(object):
    def __init__(self, config, vocab: Vocab, model_file_path=None, is_eval=False):
        encoder = Encoder(config, vocab)
        decoder = Decoder(config, vocab)
        closedbook_decoder = ClosedbookDecoder(config, vocab)  # Thêm closedbook decoder
        reduce_state = ReduceState(config, vocab)
        use_cuda = config.use_gpu
        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight
        closedbook_decoder.embedding.weight = encoder.embedding.weight  # Share embedding

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            closedbook_decoder = closedbook_decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.closedbook_decoder = closedbook_decoder  # Thêm closedbook decoder
        self.reduce_state = reduce_state
        self.vocab = vocab
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu else 'cpu')

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.closedbook_decoder.load_state_dict(state.get('closedbook_decoder_state_dict', {}), strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])

    def forward(self, src, trg, gamma=0.5):
        """
        Forward pass for hybrid training (kết hợp attention + closed-book)
        Args:
            src: (batch_size, src_len) - source sequence indices
            trg: (batch_size, trg_len) - target sequence indices
            gamma: float - weight cho closed-book loss (default: 0.5)
        Returns:
            outputs: (batch_size, trg_len, vocab_size) - logits từ attention decoder
            loss: scalar - hybrid loss = (1-γ) * L_attention + γ * L_closedbook
        """
        batch_size = src.size(0)
        src_len = src.size(1)
        trg_len = trg.size(1)
        
        # Get sequence lengths for packing
        seq_lens = []
        for i in range(batch_size):
            seq_len = (src[i] != self.vocab.pad_idx).sum().item()
            seq_lens.append(seq_len)
        seq_lens = sorted(seq_lens, reverse=True)
        
        # Encoder forward pass
        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(src, seq_lens)
        decoder_hidden = self.reduce_state(encoder_hidden)
        enc_padding_mask = (src != self.vocab.pad_idx).float()
        
        # Initialize states cho cả attention và closed-book decoder
        s_t_attn = decoder_hidden
        s_t_cb = (torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device),
                  torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device))
        c_t_attn = torch.zeros(batch_size, self.config.hidden_dim * 2, device=self.device)
        c_t_cb = torch.zeros(batch_size, self.config.hidden_dim * 2, device=self.device)
        coverage_attn = torch.zeros(batch_size, src_len, device=self.device)
        coverage_cb = torch.zeros(batch_size, self.config.max_src_len, device=self.device)
        
        # Pointer-generator setup
        extra_zeros = None
        enc_batch_extend_vocab = None
        if self.config.pointer_gen:
            enc_batch_extend_vocab = src.clone()
            extra_zeros = torch.zeros(batch_size, 1, device=self.device)
        
        # Hybrid decoding với teacher forcing
        y_t_1 = torch.full((batch_size,), self.vocab.bos_idx, dtype=torch.long, device=self.device)
        step_losses = []
        attention_outputs = []
        
        for t in range(trg_len):
            # Attention decoder step
            final_dist_attn, s_t_attn, c_t_attn, attn_dist, p_gen, coverage_attn = self.decoder(
                y_t_1, s_t_attn, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_attn, extra_zeros, enc_batch_extend_vocab, coverage_attn, t
            )
            
            # Closed-book decoder step
            final_dist_cb, s_t_cb, c_t_cb, attn_dist_cb, p_gen_cb, coverage_cb = self.closedbook_decoder(
                y_t_1, s_t_cb, c_t_cb, extra_zeros, enc_batch_extend_vocab, coverage_cb, t
            )
            
            # Tính loss cho bước t hiện tại
            target_t = trg[:, t]
            
            # Attention loss
            attn_loss_t = F.cross_entropy(final_dist_attn, target_t, ignore_index=self.vocab.pad_idx)
            
            # Closed-book loss  
            cb_loss_t = F.cross_entropy(final_dist_cb, target_t, ignore_index=self.vocab.pad_idx)
            
            # Hybrid loss cho bước t
            hybrid_loss_t = (1 - gamma) * attn_loss_t + gamma * cb_loss_t
            step_losses.append(hybrid_loss_t)
            
            # Lưu attention outputs làm chính
            attention_outputs.append(final_dist_attn)
            
            # Teacher forcing
            y_t_1 = target_t
        
        # Tính trung bình loss theo thời gian
        hybrid_loss = torch.stack(step_losses).mean()
        
        # Stack attention outputs
        outputs = torch.stack(attention_outputs, dim=1)  # (batch_size, trg_len, vocab_size)
        
        return outputs, hybrid_loss

    def forward_closed_book(self, trg):
        """
        Forward pass for Closed-Book training (không có source text)
        Args:
            trg: (batch_size, trg_len) - target sequence indices
        Returns:
            outputs: (batch_size, trg_len, vocab_size) - logits
            loss: scalar - cross entropy loss
        """
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        
        # Initialize decoder states (không có encoder)
        s_t = (torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device),
               torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device))
        c_t = torch.zeros(batch_size, self.config.hidden_dim * 2, device=self.device)
        coverage = torch.zeros(batch_size, self.config.max_src_len, device=self.device)
        
        # For pointer-generator mechanism
        extra_zeros = None
        enc_batch_extend_vocab = None
        if self.config.pointer_gen:
            # Simplified - không có source text để copy
            extra_zeros = torch.zeros(batch_size, 1, device=self.device)
        
        # Closedbook decoder forward pass (teacher forcing)
        decoder_outputs = []
        y_t_1 = torch.full((batch_size,), self.vocab.bos_idx, dtype=torch.long, device=self.device)
        
        for t in range(trg_len):
            final_dist, s_t, c_t, attn_dist, p_gen, coverage = self.closedbook_decoder(
                y_t_1, s_t, c_t, extra_zeros, enc_batch_extend_vocab, coverage, t
            )
            decoder_outputs.append(final_dist)
            
            # Teacher forcing: use ground truth as next input
            y_t_1 = trg[:, t]
        
        # Stack outputs
        outputs = torch.stack(decoder_outputs, dim=1)  # (batch_size, trg_len, vocab_size)
        
        # Calculate loss
        outputs_reshaped = outputs.view(-1, outputs.size(-1))
        trg_reshaped = trg.view(-1)
        
        # Handle invalid target values
        invalid_mask = (trg_reshaped < 0) | (trg_reshaped >= self.vocab.vocab_size)
        if invalid_mask.any():
            print("Có target không hợp lệ! Thay thế bằng pad_idx.")
        trg_reshaped = torch.where(invalid_mask, self.vocab.pad_idx, trg_reshaped)
        trg_reshaped = trg_reshaped.long()
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx)
        loss = loss_fn(outputs_reshaped, trg_reshaped)
        
        return outputs, loss

    def predict(self, src: torch.Tensor, max_len: int = None, gamma: float = 0.5) -> torch.Tensor:
        """
        Hybrid prediction - kết hợp attention và closed-book decoder
        Args:
            src: (batch_size, src_len) - source sequence indices
            max_len: maximum length of generated sequence
            gamma: float - weight cho closed-book decoder (default: 0.5)
        Returns:
            outputs: (batch_size, pred_len) - predicted sequence indices
        """
        if max_len is None:
            max_len = self.vocab.max_sentence_length + 2
        
        batch_size = src.size(0)
        src_len = src.size(1)
        
        # Encoder setup
        seq_lens = []
        for i in range(batch_size):
            seq_len = (src[i] != self.vocab.pad_idx).sum().item()
            seq_lens.append(seq_len)
        seq_lens = sorted(seq_lens, reverse=True)
        
        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(src, seq_lens)
        decoder_hidden = self.reduce_state(encoder_hidden)
        enc_padding_mask = (src != self.vocab.pad_idx).float()
        
        # Initialize states
        s_t_attn = decoder_hidden
        s_t_cb = (torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device),
                  torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device))
        c_t_attn = torch.zeros(batch_size, self.config.hidden_dim * 2, device=self.device)
        c_t_cb = torch.zeros(batch_size, self.config.hidden_dim * 2, device=self.device)
        coverage_attn = torch.zeros(batch_size, src_len, device=self.device)
        coverage_cb = torch.zeros(batch_size, self.config.max_src_len, device=self.device)
        
        # Pointer-generator setup
        extra_zeros = None
        enc_batch_extend_vocab = None
        if self.config.pointer_gen:
            enc_batch_extend_vocab = src.clone()
            extra_zeros = torch.zeros(batch_size, 1, device=self.device)
        
        # Hybrid generation
        y_t_1 = torch.full((batch_size,), self.vocab.bos_idx, dtype=torch.long, device=self.device)
        generated_tokens = []
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for t in range(max_len):
            # Attention decoder step
            final_dist_attn, s_t_attn, c_t_attn, attn_dist, p_gen, coverage_attn = self.decoder(
                y_t_1, s_t_attn, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_attn, extra_zeros, enc_batch_extend_vocab, coverage_attn, t
            )
            
            # Closed-book decoder step
            final_dist_cb, s_t_cb, c_t_cb, attn_dist_cb, p_gen_cb, coverage_cb = self.closedbook_decoder(
                y_t_1, s_t_cb, c_t_cb, extra_zeros, enc_batch_extend_vocab, coverage_cb, t
            )
            
            # Kết hợp predictions theo gamma
            combined_dist = (1 - gamma) * final_dist_attn + gamma * final_dist_cb
            
            # Get next token
            next_token = combined_dist.argmax(dim=-1)
            generated_tokens.append(next_token)
            
            # Check for EOS
            current_eos_generated = (next_token == self.vocab.eos_idx)
            finished_sequences = finished_sequences | current_eos_generated
            
            if finished_sequences.all():
                break
            
            y_t_1 = next_token
        
        outputs = torch.stack(generated_tokens, dim=1)
        return outputs

    def predict_closed_book(self, max_len: int = None) -> torch.Tensor:
        """
        Generate predictions for inference (Closed-Book - không có source text)
        Args:
            max_len: maximum length of generated sequence
        Returns:
            outputs: (batch_size, pred_len) - predicted sequence indices
        """
        if max_len is None:
            max_len = self.vocab.max_sentence_length + 2  # +2 for BOS and EOS
        
        batch_size = 1  # Simplified - single sequence generation
        
        # Initialize decoder states (không có encoder)
        s_t = (torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device),
               torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device))
        c_t = torch.zeros(batch_size, self.config.hidden_dim * 2, device=self.device)
        coverage = torch.zeros(batch_size, self.config.max_src_len, device=self.device)
        
        # For pointer-generator mechanism
        extra_zeros = None
        enc_batch_extend_vocab = None
        if self.config.pointer_gen:
            extra_zeros = torch.zeros(batch_size, 1, device=self.device)
        
        # Initialize decoder input
        y_t_1 = torch.full((batch_size,), self.vocab.bos_idx, dtype=torch.long, device=self.device)
        
        # Generate sequence
        generated_tokens = []
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for t in range(max_len):
            final_dist, s_t, c_t, attn_dist, p_gen, coverage = self.closedbook_decoder(
                y_t_1, s_t, c_t, extra_zeros, enc_batch_extend_vocab, coverage, t
            )
            
            # Get next token (greedy decoding)
            next_token = final_dist.argmax(dim=-1)
            generated_tokens.append(next_token)
            
            # Check for EOS
            current_eos_generated = (next_token == self.vocab.eos_idx)
            finished_sequences = finished_sequences | current_eos_generated
            
            # Stop if all sequences are finished
            if finished_sequences.all():
                break
            
            # Use predicted token as next input
            y_t_1 = next_token
        
        # Stack generated tokens
        outputs = torch.stack(generated_tokens, dim=1)  # (batch_size, pred_len)
        
        return outputs


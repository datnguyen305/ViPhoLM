from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import random
from builders.model_builder import META_ARCHITECTURE

class checking_cuda():
    def __init__(self,config):
        self.use_cuda = config.use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.device = self.device

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

def init_lstm_wt(config, lstm):
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

def init_linear_wt(config, linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(config,wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(config,wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(config, self.embedding.weight)
        
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(config, self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    #seq_lens should be in descending order
    def forward(self,config, input, seq_lens):
        embedded = self.embedding(input)
       
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()
        
        encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden

class ReduceState(nn.Module):
    def __init__(self,config):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(config, self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(config, self.reduce_c)

    def forward(self,config, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, config, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.attention_network = Attention(config)
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(config, self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(config, self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(config, self.out2)

    def forward(self, config, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                              enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                          enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim

        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

class ClosedBookDecoder(nn.Module):
    """ Closed-Book Decoder (LSTM không có attention) """
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True)
        self.out = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, encoder_hidden, target_seq):
        """
        encoder_hidden: Hidden state từ encoder
        target_seq: Đầu vào cho decoder (nếu training)
        """
        lstm_out, _ = self.lstm(target_seq, encoder_hidden)
        logits = self.out(lstm_out)  # (batch_size, seq_len, vocab_size)
        return F.log_softmax(logits, dim=-1)

@META_ARCHITECTURE.register()
class closedbook(nn.Module):
    def __init__(self, config, model_file_path=None, is_eval=False):
        super().__init__()
        self.encoder = Encoder(config)
        self.attn_decoder = Decoder(config)  # Decoder có attention
        self.cb_decoder = ClosedBookDecoder(config)  # Closed-Book Decoder
        self.reduce_state = ReduceState(config)

        self.gamma = config.gamma  # Trọng số giữa hai decoder

        self.use_cuda = config.use_gpu and torch.cuda.is_available()

        # Shared embedding giữa encoder và decoder
        self.attn_decoder.embedding.weight = self.encoder.embedding.weight

        if is_eval:
            self.encoder.eval()
            self.attn_decoder.eval()
            self.cb_decoder.eval()
            self.reduce_state.eval()

        if self.use_cuda:
            self.encoder.cuda()
            self.attn_decoder.cuda()
            self.cb_decoder.cuda()
            self.reduce_state.cuda()

    def forward(self, input_seq, input_lens, target_seq=None, teacher_forcing_ratio=0.5):
        """
        Forward với hai decoder (Attention-based + Closed-Book)

        input_seq: Tensor đầu vào cho encoder (batch_size, seq_len)
        input_lens: Độ dài thực của mỗi câu trong batch
        target_seq: Chuỗi đầu ra (chỉ dùng khi training)
        teacher_forcing_ratio: Xác suất teacher forcing khi training

        return:
            - combined_outputs: Dự đoán cuối cùng (kết hợp hai decoder)
            - coverage_loss: Nếu dùng coverage mechanism
        """
        # Encode input sequence
        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(input_seq, input_lens)

        # Giảm trạng thái encoder để dùng cho decoder
        decoder_hidden = self.reduce_state(encoder_hidden)

        # Nếu training, sử dụng target_seq
        if target_seq is not None:
            # Output từ Attention-based Decoder
            attn_outputs, coverage_loss = self.attn_decoder(
                input_seq, encoder_outputs, decoder_hidden, target_seq, teacher_forcing_ratio
            )

            # Output từ Closed-Book Decoder
            cb_outputs = self.cb_decoder(decoder_hidden, target_seq)

            # Kết hợp hai output với gamma
            combined_outputs = (1 - self.gamma) * attn_outputs + self.gamma * cb_outputs

            return combined_outputs, coverage_loss
        else:
            # Inference mode (chỉ dự đoán)
            attn_outputs = self.attn_decoder(input_seq, encoder_outputs, decoder_hidden)
            cb_outputs = self.cb_decoder(decoder_hidden, input_seq)

            combined_outputs = (1 - self.gamma) * attn_outputs + self.gamma * cb_outputs
            return combined_outputs

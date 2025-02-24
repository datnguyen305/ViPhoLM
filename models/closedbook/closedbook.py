from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from configs import config
from numpy import random
from builders.model_builder import META_ARCHITECTURE
from rouge_score import rouge_scorer
from torch.distributions import Categorical

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

def init_lstm_wt(lstm):
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

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    #seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)
       
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()
        
        encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden

class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
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

class Pointer_Decoder(nn.Module):
    def __init__(self):
        super(Pointer_Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
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

            p_attn = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            p_attn = vocab_dist

        return p_attn, s_t, c_t, attn_dist, p_gen, coverage

class ClosedBookDecoder(nn.Module):
    def __init__(self):
        super(ClosedBookDecoder, self).__init__()
        # Closed-book decoder (Unidirectional LSTM without attention or pointer layer)
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        self.out1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

        self.p_cbdec = nn.Linear(config.hidden_dim, 1)  # Probability of closed-book decoder

    def forward(self, y_t_1, s_t_1):
        y_t_1_embd = self.embedding(y_t_1)
        lstm_out, s_t = self.lstm(y_t_1_embd.unsqueeze(1), s_t_1)

        output = self.out1(lstm_out.view(-1, config.hidden_dim))
        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        p_cbdec = torch.sigmoid(self.p_cbdec(lstm_out.view(-1, config.hidden_dim)))  # Compute Pcbdec

        return vocab_dist, s_t, p_cbdec

def loss_fn(logits_attn, logits_cbdec, targets, gamma):

    probs_attn = F.softmax(logits_attn, dim=-1)  
    probs_cbdec = F.softmax(logits_cbdec, dim=-1) 

    target_probs_attn = probs_attn.gather(2, targets.unsqueeze(-1)).squeeze(-1) 
    target_probs_cbdec = probs_cbdec.gather(2, targets.unsqueeze(-1)).squeeze(-1)  

    log_probs_attn = torch.log(target_probs_attn + 1e-9)  
    log_probs_cbdec = torch.log(target_probs_cbdec + 1e-9)  

    loss = -((1 - gamma) * log_probs_attn + gamma * log_probs_cbdec)

    return loss.mean()


@META_ARCHITECTURE.register()
class closedbook(object):
    def __init__(self, config, model_file_path=None, is_eval=False):
        super(closedbook, self).__init__()

        self.config = config
        self.device = torch.device("cuda" if config["model"]["device"] == "cuda" and torch.cuda.is_available() else "cpu")

        # Tạo encoder và decoder
        self.encoder = Encoder(config["model"]["encoder"])
        self.pointer_decoder = Pointer_Decoder(config["model"]["decoder"], config["model"]["pointer_gen"])
        self.closed_book_decoder = ClosedBookDecoder(config["model"]["decoder"])
        self.reduce_state = ReduceState(config["model"]["encoder"]["hidden_size"])

        # Chia sẻ embedding giữa encoder và decoder
        self.pointer_decoder.embedding.weight = self.encoder.embedding.weight
        self.closed_book_decoder.embedding.weight = self.encoder.embedding.weight

        if is_eval:
            self.encoder.eval()
            self.pointer_decoder.eval()
            self.closed_book_decoder.eval()
            self.reduce_state.eval()

        self.to(self.device)

    def forward(self, input_ids, tokenizer, max_length=100):
        """
        Forward cho RL: Sinh greedy summary và sampled summary.
        """
        encoder_outputs, encoder_hidden = self.encoder(input_ids)

        # 1️⃣ Greedy decoding (ŷ)
        greedy_summary = []
        decoder_input = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(self.device)
        decoder_hidden = encoder_hidden
        for _ in range(max_length):
            final_dist, _, _, _, _, _, _ = self.pointer_decoder(
                decoder_input, decoder_hidden, encoder_outputs, None, None, None, None, _
            )
            logits_cbdec, _, _ = self.closed_book_decoder(decoder_input, decoder_hidden)

            gamma = self.config["model"]["pointer_gen"]
            logits = (1 - gamma) * torch.log(final_dist + 1e-9) + gamma * logits_cbdec

            next_token = torch.argmax(logits, dim=-1).item()
            if next_token == tokenizer.eos_token_id:
                break

            greedy_summary.append(next_token)
            decoder_input = torch.tensor([[next_token]], dtype=torch.long).to(self.device)

        # 2️⃣ Sampling decoding (y_s)
        sampled_summary = []
        log_probs = []
        decoder_input = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(self.device)
        decoder_hidden = encoder_hidden
        for _ in range(max_length):
            final_dist, _, _, _, _, _, _ = self.pointer_decoder(
                decoder_input, decoder_hidden, encoder_outputs, None, None, None, None, _
            )
            logits_cbdec, _, _ = self.closed_book_decoder(decoder_input, decoder_hidden)

            gamma = self.config["model"]["pointer_gen"]
            logits = (1 - gamma) * torch.log(final_dist + 1e-9) + gamma * logits_cbdec

            dist = Categorical(logits.exp())  # Phân phối xác suất
            sampled_token = dist.sample()
            log_prob = dist.log_prob(sampled_token)

            sampled_summary.append(sampled_token.item())
            log_probs.append(log_prob)

            if sampled_token.item() == tokenizer.eos_token_id:
                break

            decoder_input = sampled_token.unsqueeze(0)

        return greedy_summary, sampled_summary, log_probs

    def compute_reward(self, predicted_text, reference_text):
        """ Tính reward bằng ROUGE-L """
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference_text, predicted_text)
        return scores['rougeL'].fmeasure

    def rl_loss(self, log_probs, r_greedy, r_sampled):
        """ Tính loss RL """
        reward_diff = r_greedy - r_sampled
        loss = -sum(log_prob * reward_diff for log_prob in log_probs)
        return loss / len(log_probs)

    def train_step(self, optimizer, input_text, reference_text, tokenizer):
        """ Huấn luyện với RL + XE """
        optimizer.zero_grad()
        
        greedy_summary, sampled_summary, log_probs = self.forward(input_text, tokenizer)
        
        greedy_text = tokenizer.decode(greedy_summary, skip_special_tokens=True)
        sampled_text = tokenizer.decode(sampled_summary, skip_special_tokens=True)

        r_greedy = self.compute_reward(greedy_text, reference_text)
        r_sampled = self.compute_reward(sampled_text, reference_text)

        loss_rl = self.rl_loss(log_probs, r_greedy, r_sampled)
        loss_xe = loss_fn(...)  # Cross-Entropy loss (cần triển khai)

        lambda_rl = 0.8  # Hệ số RL
        loss = lambda_rl * loss_rl + (1 - lambda_rl) * loss_xe

        loss.backward()
        optimizer.step()

        return loss.item(), r_sampled, r_greedy

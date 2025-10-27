import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import Counter
from cytoolz import concat

# ==============================================================================
# Helper functions from original util.py
# ==============================================================================

def len_mask(lens, device):
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.ByteTensor(batch_size, max_len).to(device).fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l].fill_(1)
    return mask

def sequence_mean(sequence, seq_lens, dim=1):
    if seq_lens:
        sum_ = torch.sum(sequence, dim=dim, keepdim=False)
        mean = torch.stack([s/l for s, l in zip(sum_, seq_lens)], dim=0)
    else:
        mean = torch.mean(sequence, dim=dim, keepdim=False)
    return mean

def reorder_sequence(sequence_emb, order, batch_first=False):
    batch_dim = 0 if batch_first else 1
    order = torch.LongTensor(order).to(sequence_emb.device)
    return sequence_emb.index_select(index=order, dim=batch_dim)

def reorder_lstm_states(lstm_states, order):
    order = torch.LongTensor(order).to(lstm_states[0].device)
    return (lstm_states[0].index_select(index=order, dim=1),
            lstm_states[1].index_select(index=order, dim=1))

# ==============================================================================
# Helper function from original myownutils.py
# ==============================================================================

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)

# ==============================================================================
# Attention functions from original attention.py
# ==============================================================================

def prob_normalize(score, mask):
    score = score.masked_fill(mask == 0, -1e18)
    return F.softmax(score, dim=-1)

def step_attention(query, key, value, mem_mask=None):
    score = query.unsqueeze(-2).matmul(key.transpose(1, 2))
    norm_score = prob_normalize(score, mem_mask) if mem_mask is not None else F.softmax(score, dim=-1)
    output = norm_score.matmul(value)
    return output.squeeze(-2), norm_score.squeeze(-2)

# ==============================================================================
# Functional LSTM Encoder from original rnn.py
# ==============================================================================

def lstm_encoder(sequence, lstm, seq_lens=None, init_states=None, embedding=None):
    if embedding:
        sequence = embedding(sequence)
    
    if not lstm.batch_first:
        sequence = sequence.transpose(0, 1)

    if seq_lens:
        sort_ind = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i], reverse=True)
        seq_lens_sorted = [seq_lens[i] for i in sort_ind]
        sequence = reorder_sequence(sequence, sort_ind, lstm.batch_first)

        packed_seq = nn.utils.rnn.pack_padded_sequence(sequence, seq_lens_sorted, batch_first=lstm.batch_first)
        packed_out, final_states = lstm(packed_seq, init_states)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=lstm.batch_first)

        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(len(seq_lens))]
        lstm_out = reorder_sequence(lstm_out, reorder_ind, lstm.batch_first)
        final_states = reorder_lstm_states(final_states, reorder_ind)
    else:
        lstm_out, final_states = lstm(sequence, init_states)

    return lstm_out, final_states

# ==============================================================================
# Beam Search logic from original beam_search.py
# ==============================================================================

class _Hypothesis(object):
    def __init__(self, sequence, logprob, hists, attns=None):
        self.sequence = sequence
        self.logprob = logprob
        self.hists = hists
        self.attns = attns if attns is not None else []

    def extend(self, token, logprob, hists, attn=None):
        return _Hypothesis(
            self.sequence + [token], self.logprob + logprob, hists, self.attns + ([attn] if attn else [])
        )
    
    @property
    def latest_token(self):
        return self.sequence[-1]

    def __lt__(self, other):
        return self.logprob / len(self.sequence) < other.logprob / len(other.sequence)
    
def init_beam(start, hists):
    """ get a initial beam to start beam search"""
    return [_Hypothesis([start], 0, hists)]


def create_beam(tok, lp, hists):
    """ initailiza a beam with top k token"""
    k = tok.size(0)
    return [_Hypothesis([tok[i].item()], lp[i].item(), hists)
            for i in range(k)]


def pack_beam(hyps, device):
    """pack a list of hypothesis to decoder input batches"""
    token = torch.LongTensor([h.sequence[-1] for h in hyps])

    hists = tuple(torch.stack([hyp.hists[i] for hyp in hyps], dim=d)
                  for i, d in enumerate([1, 1, 0]))
    token = token.to(device)
    states = ((hists[0], hists[1]), hists[2])
    return token, states


def next_search_beam(beam, beam_size, finished,
                     end, topk, lp, hists, attn=None, diverse=1.0):
    """generate the next beam(K-best hyps)"""
    topks, lps, hists_list, attns = _unpack_topk(topk, lp, hists, attn)
    hyps_lists = [h.extend_k(topks[i], lps[i],
                             hists_list[i], attns[i], diverse)
                  for i, h in enumerate(beam)]
    hyps = list(concat(hyps_lists))
    finished, beam = _clean_beam(finished, hyps, end, beam_size)

    return finished, beam

def next_search_beam_cnn(beam, beam_size, finished,
                     end, topk, lp, hists, attn=None, diverse=1.0):
    """generate the next beam(K-best hyps)"""
    topks, lps, hists_list, attns = _unpack_topk(topk, lp, hists, attn)
    hyps_lists = [h.extend_k(topks[i], lps[i],
                             hists_list[i], attns[i], diverse)
                  for i, h in enumerate(beam)]
    hyps = list(concat(hyps_lists))
    finished, beam = _clean_beam_cnn(finished, hyps, end, beam_size)

    return finished, beam


def best_sequence(finished, beam=None):
    """ return the sequence with the highest prob(normalized by length)"""
    if beam is None:  # not empty
        best_beam = finished[0]
    else:
        if finished and beam[0] < finished[0]:
            best_beam = finished[0]
        else:
            best_beam = beam[0]

    best_seq = best_beam.sequence[1:]
    if best_beam.attns:
        return best_seq, best_beam.attns
    else:
        return best_seq


def _unpack_topk(topk, lp, hists, attn=None):
    """unpack the decoder output"""
    beam, _ = topk.size()
    topks = [t for t in topk]
    lps = [l for l in lp]
    k_hists = [(hists[0][:, i, :], hists[1][:, i, :], hists[2][i, :])
               for i in range(beam)]

    if attn is None:
        return topks, lps, k_hists
    else:
        attns = [attn[i] for i in range(beam)]
        return topks, lps, k_hists, attns

def length_wu(cur_len, alpha=0.):
    """GNMT length re-ranking score.
    See "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """
    return ((5 + cur_len) / 6.0) ** alpha


def coverage_summary(cov, beta=0.):
    """Our summary penalty."""
    penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(-1)
    penalty -= cov.size(-1)
    return beta * penalty

def _clean_beam(finished, beam, end_tok, beam_size, remove_tri=True):
    """ remove completed sequence from beam """
    new_beam = []
    # for h in sorted(beam, reverse=True,
    #                 key=lambda h: h.logprob/len(h.sequence)):
    for h in sorted(beam, reverse=True,
                    key=lambda h: h.logprob / length_wu(len(h.sequence), alpha=0.9) - coverage_summary(h.coverage, beta=5)):
    # for h in sorted(beam, reverse=True,
    #                 key=lambda h: h.logprob / length_wu(len(h.sequence), alpha=0.9)):
        if remove_tri and _has_repeat_tri(h.sequence):
            h.logprob = -1e9
        if h.sequence[-1] == end_tok:
            finished_hyp = _Hypothesis(h.sequence[:-1], # remove EOS
                                       h.logprob, h.hists, h.attns, h.coverage)
            finished.append(finished_hyp)
        else:
            new_beam.append(h)
        if len(new_beam) == beam_size:
            break
    else:
        # ensure beam size
        while len(new_beam) < beam_size:
            new_beam.append(new_beam[0])

    finished = sorted(finished, reverse=True,
                      key=lambda h: h.logprob/len(h.sequence))
    return finished, new_beam

def _clean_beam_cnn(finished, beam, end_tok, beam_size, remove_tri=True):
    """ remove completed sequence from beam """
    new_beam = []
    # for h in sorted(beam, reverse=True,
    #                 key=lambda h: h.logprob/len(h.sequence)):
    # for h in sorted(beam, reverse=True,
    #                 key=lambda h: h.logprob / length_wu(len(h.sequence), alpha=0.9) - coverage_summary(h.coverage, beta=5)):
    for h in sorted(beam, reverse=True,
                    key=lambda h: h.logprob / len(h.sequence)):
        if remove_tri and _has_repeat_tri(h.sequence):
            h.logprob = -1e9
            continue
        if h.sequence[-1] == end_tok:
            finished_hyp = _Hypothesis(h.sequence[:-1], # remove EOS
                                       h.logprob, h.hists, h.attns)
            finished.append(finished_hyp)
        else:
            new_beam.append(h)
        if len(new_beam) == beam_size:
            break
    else:
        # ensure beam size
        while len(new_beam) < beam_size:
            new_beam.append(new_beam[0])

    finished = sorted(finished, reverse=True,
                      key=lambda h: h.logprob/len(h.sequence))
    return finished, new_beam


def _has_repeat_tri(grams):
    tri_grams = [tuple(grams[i:i+3]) for i in range(len(grams)-2)]
    cnt = Counter(tri_grams)
    return not all((cnt[g] <= 1 for g in cnt))
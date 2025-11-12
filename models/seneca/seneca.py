# seneca.py
from typing import List, Tuple, Dict, Any, Optional
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import Counter
from cytoolz import concat
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab


# -----------------------
# 1) Utilities
# -----------------------

INI = 1e-2


def len_mask(lens, device):
    """Return boolean mask [B, max_len] where True indicates valid positions."""
    if isinstance(lens, torch.Tensor):
        lens = lens.tolist()
    if not isinstance(lens, (list, tuple)) or len(lens) == 0:
        return torch.zeros((0, 0), dtype=torch.bool, device=device)
    max_len = max(lens)
    batch_size = len(lens)
    arange = torch.arange(max_len, device=device).expand(batch_size, max_len)
    lens_tensor = torch.tensor(lens, device=device).unsqueeze(1)
    return arange < lens_tensor


def sequence_mean(sequence: torch.Tensor, seq_lens: Optional[List[int]] = None, dim: int = 1):
    """Mean pooling over valid positions using seq_lens."""
    if seq_lens is None:
        return sequence.mean(dim=dim)
    mask = len_mask(seq_lens, sequence.device).unsqueeze(-1).float()
    summed = (sequence * mask).sum(dim=dim)
    lens_tensor = torch.tensor(seq_lens, device=sequence.device).unsqueeze(-1).float().clamp(min=1.0)
    return summed / lens_tensor


def prob_normalize(score: torch.Tensor, mask: torch.Tensor):
    """Masked softmax with safety when entire row masked."""
    # score: [..., L], mask: [..., L] boolean
    score = score.masked_fill(~mask, -1e18)
    # handle fully masked rows by setting them to zeros (uniform tiny logits)
    all_masked = (~mask).all(dim=-1)
    if all_masked.any():
        # set those rows to zero so softmax returns uniform distribution
        score = score.masked_fill(all_masked.unsqueeze(-1), 0.0)
    return F.softmax(score, dim=-1)


def step_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mem_mask: Optional[torch.Tensor] = None):
    """
    Compute single-step attention.
    query: [B, Dq]
    key: [B, L, Dk]  (we assume Dq == Dk)
    value: [B, L, Dv]
    mem_mask: [B, 1, L] or [B, L]
    Returns: (context [B, Dv], norm_score [B, L])
    """
    # [B, 1, L]
    score = torch.bmm(query.unsqueeze(1), key.transpose(1, 2))
    if mem_mask is not None:
        m = mem_mask
        if m.dim() == 2:
            m = m.unsqueeze(1)
        norm = prob_normalize(score, m)
    else:
        norm = F.softmax(score, dim=-1)
    ctx = torch.bmm(norm, value)  # [B, 1, Dv]
    return ctx.squeeze(1), norm.squeeze(1)


def reorder_sequence(sequence_emb: torch.Tensor, order: List[int], batch_first: bool = True):
    dim = 0 if batch_first else 1
    order_t = torch.LongTensor(order).to(sequence_emb.device)
    return sequence_emb.index_select(dim, order_t)


def reorder_lstm_states(lstm_states: Tuple[torch.Tensor, torch.Tensor], order: List[int]):
    order_t = torch.LongTensor(order).to(lstm_states[0].device)
    h = lstm_states[0].index_select(1, order_t)
    c = lstm_states[1].index_select(1, order_t)
    return (h, c)


def lstm_encoder(sequence: torch.Tensor, lstm: nn.LSTM, seq_lens: Optional[List[int]] = None,
                 init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 embedding: Optional[nn.Embedding] = None):
    """
    sequence: [B, L] token ids or [B, L, D_emb] embeddings
    returns: (lstm_out [B,L,D], (h, c))
    """
    if embedding is not None:
        sequence = embedding(sequence)
    if not lstm.batch_first:
        sequence = sequence.transpose(0, 1)
    if seq_lens is not None:
        if isinstance(seq_lens, torch.Tensor):
            seq_lens = seq_lens.cpu().tolist()
        # sort
        sort_ind = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i], reverse=True)
        seq_lens_sorted = [seq_lens[i] for i in sort_ind]
        sequence = reorder_sequence(sequence, sort_ind, batch_first=lstm.batch_first)
        packed = nn.utils.rnn.pack_padded_sequence(sequence, seq_lens_sorted, batch_first=lstm.batch_first)
        packed_out, final_states = lstm(packed, init_states)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=lstm.batch_first)
        # restore order
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(len(sort_ind))]
        out = reorder_sequence(out, reorder_ind, batch_first=lstm.batch_first)
        final_states = reorder_lstm_states(final_states, reorder_ind)
    else:
        out, final_states = lstm(sequence, init_states)
    return out, final_states


class _Hypothesis:
    def __init__(self, sequence: List[int], logprob: float, hists: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 attns: Optional[List[torch.Tensor]] = None, coverage: Optional[torch.Tensor] = None):
        self.sequence = sequence
        self.logprob = logprob
        self.hists = hists  # (h, c, prev_out)
        self.attns = attns if attns is not None else []
        self.coverage = coverage if coverage is not None else torch.zeros(0)

    def extend(self, token: int, logprob: float, hists, attn: Optional[torch.Tensor] = None):
        new_attns = self.attns + ([attn] if attn is not None else [])
        new_cov = self.coverage + attn if (attn is not None and self.coverage.numel() != 0) else self.coverage if self.coverage.numel() != 0 else (attn if attn is not None else self.coverage)
        return _Hypothesis(self.sequence + [token], self.logprob + logprob, hists, new_attns, new_cov)

    def extend_k(self, topk_ids, topk_logprobs, hists, attns=None, diverse=1.0):
        hyps = []
        attns = attns or [None] * len(topk_ids)
        for tok, lp, at in zip(topk_ids, topk_logprobs, attns):
            hyps.append(self.extend(int(tok.item()), float(lp.item()), hists, at))
        return hyps

    @property
    def latest_token(self):
        return self.sequence[-1]

    def __lt__(self, other):
        return (self.logprob / len(self.sequence)) < (other.logprob / len(other.sequence))


def init_beam(start: int, hists):
    return [_Hypothesis([start], 0.0, hists)]

def next_search_beam(beam, beam_size, finished, end_tok, topk_idx, topk_logprob, hists, attn=None, diverse=1.0):
    hyps_lists = [h.extend_k(topk_idx[i], topk_logprob[i], hists, attn=None, diverse=diverse) for i, h in enumerate(beam)]
    hyps = list(concat(hyps_lists))
    finished, beam = _clean_beam(finished, hyps, end_tok, beam_size)
    return finished, beam


def pack_beam(hyps: List[_Hypothesis], device: torch.device):
    """Pack beam into token tensor and stacked states compatible with decoder."""
    token = torch.LongTensor([h.latest_token for h in hyps]).to(device)
    # hists: (h, c, prev_out)
    # each h.c: [num_layers, B, hidden] originally; here hyp.hists[0] shape [num_layers, hidden] per hypothesis
    # We'll stack across batch axis (dim=1) for h,c and dim=0 for prev_out
    h_stack = torch.stack([h.hists[0] for h in hyps], dim=1)  # [num_layers, beam, hidden]
    c_stack = torch.stack([h.hists[1] for h in hyps], dim=1)
    prev_out_stack = torch.stack([h.hists[2] for h in hyps], dim=0)  # [beam, emb_dim]
    states = ((h_stack, c_stack), prev_out_stack)
    return token, states


def _clean_beam(finished: List[_Hypothesis], beam: List[_Hypothesis], end_tok: int, beam_size: int, remove_tri: bool = True):
    # sort by normalized score with length/coverage penalties
    sorted_beam = sorted(beam, reverse=True, key=lambda h: h.logprob / max(1, len(h.sequence)))
    new_beam = []
    for h in sorted_beam:
        if remove_tri and _has_repeat_tri(h.sequence):
            h.logprob = -1e9
        if h.latest_token == end_tok:
            finished_hyp = _Hypothesis(h.sequence[:-1], h.logprob, h.hists, h.attns, h.coverage)
            finished.append(finished_hyp)
        else:
            new_beam.append(h)
        if len(new_beam) >= beam_size:
            break
    if not new_beam and sorted_beam:
        new_beam = [sorted_beam[0]]
    finished = sorted(finished, reverse=True, key=lambda h: h.logprob / max(1, len(h.sequence)))
    return finished, new_beam


def _has_repeat_tri(grams: List[int]):
    if len(grams) < 3:
        return False
    tri = [tuple(grams[i:i+3]) for i in range(len(grams) - 2)]
    cnt = Counter(tri)
    return any(c > 1 for c in cnt.values())


class ConvEncoder(nn.Module):
    """
    Multi-kernel 1D conv encoder.
    Input: [B, L, D_emb]
    Output: [B, L', (n_hidden * n_kernels)]
    """

    def __init__(self, emb_dim: int, n_hidden: int, kernel_sizes: List[int], dropout: float = 0.1):
        super().__init__()
        self._convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim, out_channels=n_hidden, kernel_size=k, padding=(k - 1) // 2)
            for k in kernel_sizes
        ])
        self._dropout = dropout
        self._n_kernels = len(kernel_sizes)
        self._n_hidden = n_hidden

    def forward(self, emb_input: torch.Tensor):
        # emb_input: [B, L, D] -> [B, D, L]
        x = emb_input.transpose(1, 2)
        x = F.dropout(x, p=self._dropout, training=self.training)
        outs = [F.relu(conv(x)) for conv in self._convs]
        min_len = min(o.size(2) for o in outs)
        outs = [o[:, :, :min_len] for o in outs]
        cat = torch.cat(outs, dim=1)  # [B, n_hidden * n_kernels, L]
        return cat.transpose(1, 2)  # [B, L, n_hidden * n_kernels]


class MultiLayerLSTMCells(nn.Module):
    """Stacked LSTMCell layers for decoder unroll and beam search compatibility."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self._dropout = dropout
        cells = []
        for i in range(num_layers):
            insz = input_size if i == 0 else hidden_size
            cells.append(nn.LSTMCell(insz, hidden_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, input_: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]):
        """
        input_: [B, input_size]
        states: (h_prev, c_prev): each [num_layers, B, hidden]
        returns: (h_next, c_next): same shapes
        """
        h_prev, c_prev = states
        h_next_l = []
        c_next_l = []
        layer_input = input_
        for i, cell in enumerate(self.cells):
            h_i, c_i = cell(layer_input, (h_prev[i], c_prev[i]))
            h_next_l.append(h_i)
            c_next_l.append(c_i)
            if i < len(self.cells) - 1:
                layer_input = F.dropout(h_i, p=self._dropout, training=self.training)
            else:
                layer_input = h_i
        h_next = torch.stack(h_next_l, dim=0)
        c_next = torch.stack(c_next_l, dim=0)
        return h_next, c_next

    @staticmethod
    def convert(lstm_module: nn.LSTM):
        """Optional: convert weights from nn.LSTM into MultiLayerLSTMCells."""
        new = MultiLayerLSTMCells(lstm_module.input_size, lstm_module.hidden_size, lstm_module.num_layers, getattr(lstm_module, 'dropout', 0.0))
        for i, cell in enumerate(new.cells):
            cell.weight_ih.data.copy_(getattr(lstm_module, f'weight_ih_l{i}'))
            cell.weight_hh.data.copy_(getattr(lstm_module, f'weight_hh_l{i}'))
            cell.bias_ih.data.copy_(getattr(lstm_module, f'bias_ih_l{i}'))
            cell.bias_hh.data.copy_(getattr(lstm_module, f'bias_hh_l{i}'))
        return new


class _CopyLinear(nn.Module):
    """Compute copy gate logits: v_c^T c + v_s^T s + v_i^T x + b"""

    def __init__(self, context_dim: int, state_dim: int, input_dim: int, bias: bool = True):
        super().__init__()
        self._v_c = nn.Parameter(torch.empty(context_dim))
        self._v_s = nn.Parameter(torch.empty(state_dim))
        self._v_i = nn.Parameter(torch.empty(input_dim))
        init = torch.nn.init.uniform_
        init(self._v_c, -INI, INI)
        init(self._v_s, -INI, INI)
        init(self._v_i, -INI, INI)
        self._b = nn.Parameter(torch.zeros(1)) if bias else None

    def forward(self, context: torch.Tensor, state: torch.Tensor, input_: torch.Tensor):
        # context: [B, context_dim], state: [B, state_dim], input_: [B, input_dim]
        out = (context @ self._v_c.unsqueeze(-1) +
               state @ self._v_s.unsqueeze(-1) +
               input_ @ self._v_i.unsqueeze(-1))
        if self._b is not None:
            out = out + self._b
        return out  # [B, 1]


class AttentionalLSTMDecoder(nn.Module):
    """Base attentional decoder; not used directly in training loop here but helpful."""

    def __init__(self, embedding: nn.Embedding, lstm_cells: MultiLayerLSTMCells, attn_w: torch.Tensor, projection: nn.Module):
        super().__init__()
        self._embedding = embedding
        self._lstm = lstm_cells
        self._attn_w = attn_w
        self._projection = projection


class CopyLSTMDecoder(AttentionalLSTMDecoder):
    """
    LSTM decoder with copy mechanism (See et al., 2017 style).
    Supports both step-by-step and teacher forcing decoding.
    """

    def __init__(self, copy_linear: _CopyLinear, embedding: nn.Embedding,
                 lstm_cells: MultiLayerLSTMCells, attn_w: torch.Tensor, projection: nn.Module):
        super().__init__(embedding, lstm_cells, attn_w, projection)
        self._copy = copy_linear

    def _step(self, tok: torch.Tensor, states, attention):
        """
        Perform one decoding step.
        """
        (prev_h, prev_c), prev_out = states
        emb = self._embedding(tok).squeeze(1)  # [B, emb]
        lstm_in = torch.cat([emb, prev_out], dim=-1)
        new_h, new_c = self._lstm(lstm_in, (prev_h, prev_c))
        lstm_out = new_h[-1]  # [B, hidden]

        enc_mem, enc_proj, mask, extend_art, extend_vsize = attention
        query = lstm_out @ self._attn_w  # [B, attn_dim]
        ctx, score = step_attention(query, enc_proj, enc_mem, mask)  # ctx: [B, enc_dim]

        dec_out = self._projection(torch.cat([lstm_out, ctx], dim=-1))  # [B, emb_dim]
        print(lstm_out.shape, ctx.shape)


        # Generation probability
        logits = dec_out @ self._embedding.weight.T  # [B, vocab_size]
        if extend_vsize > logits.size(1):
            pad = torch.full((logits.size(0), extend_vsize - logits.size(1)), -1e8, device=logits.device)
            logits = torch.cat([logits, pad], dim=1)
        gen_prob = F.softmax(logits, dim=-1)

        # Copy gate
        copy_gate = torch.sigmoid(self._copy(ctx, lstm_out, emb))  # [B, 1]

        # Copy distribution
        copy_vals = (score * copy_gate.squeeze(-1))
        add_tensor = torch.zeros_like(gen_prob)
        add_tensor.scatter_add_(1, extend_art.long(), copy_vals)
        final_prob = (1 - copy_gate) * gen_prob + add_tensor
        log_prob = torch.log(final_prob + 1e-12)

        new_states = ((new_h, new_c), dec_out)
        return log_prob, new_states, score

    def forward(self, attention, init_states, abstract):
        """
        Teacher forcing decoding for training.
        Args:
            attention: (enc_mem, enc_proj, mask, extend_art, extend_vsize)
            init_states: ((h, c), prev_out)
            abstract: [B, T] target sequence
        Returns:
            log_probs: [B, T, extend_vsize]
        """
        B, T = abstract.size()
        device = abstract.device
        states = init_states
        outputs = []

        for t in range(T):
            tok = abstract[:, t:t+1]
            logp, states, _ = self._step(tok, states, attention)
            outputs.append(logp)

        return torch.stack(outputs, dim=1)  # [B, T, extend_vsize]

    def topk_step(self, tok, states, attention, beam_size, force_not_stop=False):
        log_prob, new_states, score = self._step(tok, states, attention)
        topk_logprob, topk_idx = torch.topk(log_prob, beam_size, dim=-1)
        return topk_idx, topk_logprob, new_states, score



class LSTMPointerNet(nn.Module):
    """
    Pointer network decoder with entity attention.
    Returns scores over sentences (not softmaxed).
    """

    def __init__(self, input_dim: int, n_hidden: int, n_layer: int, dropout: float, n_hop: int, side_dim: int):
        super().__init__()
        self._lstm = nn.LSTM(input_dim, n_hidden, num_layers=n_layer, batch_first=True, dropout=dropout if n_layer > 1 else 0.0)
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        self.side_wm = nn.Parameter(torch.Tensor(side_dim, n_hidden))
        self.side_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.side_v = nn.Parameter(torch.Tensor(n_hidden))
        self._attn_ws = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        for p in [self._attn_wm, self._attn_wq, self.side_wm, self.side_wq, self._attn_ws]:
            nn.init.xavier_normal_(p)
        for p in [self._attn_v, self.side_v]:
            nn.init.uniform_(p, -INI, INI)

    def forward(self, sent_mem: torch.Tensor, entity_mem: torch.Tensor, ptr_in: torch.Tensor, sent_nums: List[int], entity_nums: List[int]):
        # ptr_in: [B, T, input_dim]
        query, _ = self._lstm(ptr_in)  # [B, T, n_hidden]
        # entity attention
        side_feat = entity_mem @ self.side_wm  # [B, Ne, n_hidden]
        entity_ctx = self._attention(side_feat, query, self.side_v, self.side_wq, entity_nums)  # [B, T, side_dim]
        # sentence attention (with entity ctx)
        sent_feat = sent_mem @ self._attn_wm  # [B, Ns, n_hidden]
        score = self._attn_with_side(sent_feat, query, entity_ctx, self._attn_v, self._attn_wq, self._attn_ws, sent_nums)  # [B, T, Ns]
        return score

    @staticmethod
    def _attention(mem, query, v, w, sizes):
        score = torch.tanh(mem.unsqueeze(1) + query.matmul(w).unsqueeze(2)).matmul(v.unsqueeze(-1)).squeeze(-1)
        mask = len_mask(sizes, score.device).unsqueeze(1)
        score = score.masked_fill(~mask, -1e9)
        attn = F.softmax(score, dim=-1)
        return attn.matmul(mem)

    @staticmethod
    def _attn_with_side(mem, query, ctx, v, wq, ws, sizes):
        s = query.matmul(wq).unsqueeze(2)
        e = ctx.matmul(ws).unsqueeze(2)
        score = torch.tanh(mem.unsqueeze(1) + s + e).matmul(v.unsqueeze(-1)).squeeze(-1)
        mask = len_mask(sizes, score.device).unsqueeze(1)
        score = score.masked_fill(~mask, -1e9)
        return score


class EntityAwareExtractor(nn.Module):
    """
    Entity-aware sentence extractor using ConvEncoder + BiLSTM + LSTMPointerNet.
    """

    def __init__(self, vocab_size: int, emb_dim: int, conv_hidden: int, lstm_hidden: int,
                 lstm_layer: int, bidirectional: bool, n_hop: int, dropout: float):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.sent_conv_encoder = ConvEncoder(emb_dim=emb_dim, n_hidden=conv_hidden, kernel_sizes=[3, 4, 5], dropout=dropout)
        self.sent_lstm_encoder = nn.LSTM(input_size=3 * conv_hidden, hidden_size=lstm_hidden, num_layers=lstm_layer,
                                         bidirectional=bidirectional, batch_first=True, dropout=dropout if lstm_layer > 1 else 0.0)
        self.entity_conv_encoder = ConvEncoder(emb_dim=emb_dim, n_hidden=conv_hidden, kernel_sizes=[2, 3, 4], dropout=dropout)
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        entity_dim = 3 * conv_hidden
        self.pointer_decoder = LSTMPointerNet(input_dim=enc_out_dim, n_hidden=lstm_hidden, n_layer=lstm_layer, dropout=dropout, n_hop=n_hop, side_dim=entity_dim)

    def _encode_sentences(self, article_sents: torch.Tensor, sent_nums: List[int]):
        B, max_sents, max_words = article_sents.size()
        sents_emb = self._embedding(article_sents)  # [B, Ns, Nw, D]
        sents_flat = sents_emb.view(B * max_sents, max_words, -1)  # [B*Ns, Nw, D]
        conv_sents = self.sent_conv_encoder(sents_flat)  # [B*Ns, L', 3*conv_hidden]
        # max-over-time pooling
        sent_vecs = conv_sents.max(dim=1).values  # [B*Ns, feat]
        sent_vecs = sent_vecs.view(B, max_sents, -1)
        if isinstance(sent_nums, torch.Tensor):
            sent_nums = sent_nums.cpu().tolist()
        packed = nn.utils.rnn.pack_padded_sequence(sent_vecs, sent_nums, batch_first=True, enforce_sorted=False)
        sent_mem, _ = self.sent_lstm_encoder(packed)
        sent_mem, _ = nn.utils.rnn.pad_packed_sequence(sent_mem, batch_first=True)
        return sent_mem  # [B, Ns, enc_out_dim]

    def _encode_entities(self, clusters: torch.Tensor, cluster_nums: List[int]):
        B, max_clusters, max_mentions = clusters.size()
        ent_emb = self._embedding(clusters)  # [B, Ne, Nm, D]
        ent_flat = ent_emb.view(B * max_clusters, max_mentions, -1)
        conv_ents = self.entity_conv_encoder(ent_flat)  # [B*Ne, L', feat]
        entity_vecs = conv_ents.max(dim=1).values
        entity_mem = entity_vecs.view(B, max_clusters, -1)
        return entity_mem

    def forward(self, article_sents: torch.Tensor, sent_nums: torch.Tensor, clusters: torch.Tensor, cluster_nums: torch.Tensor, target_indices: torch.Tensor):
        sent_mem = self._encode_sentences(article_sents, sent_nums)
        entity_mem = self._encode_entities(clusters, cluster_nums)
        B, T = target_indices.size()
        D = sent_mem.size(-1)
        tgt = target_indices.clamp(0, sent_mem.size(1) - 1)
        ptr_in = torch.gather(sent_mem, dim=1, index=tgt.unsqueeze(-1).expand(B, T, D))
        sent_nums_list = sent_nums.cpu().tolist() if isinstance(sent_nums, torch.Tensor) else sent_nums
        cluster_nums_list = cluster_nums.cpu().tolist() if isinstance(cluster_nums, torch.Tensor) else cluster_nums
        scores = self.pointer_decoder(sent_mem=sent_mem, entity_mem=entity_mem, ptr_in=ptr_in, sent_nums=sent_nums_list, entity_nums=cluster_nums_list)
        return scores

    @torch.no_grad()
    def extract(self, article_sents: torch.Tensor, sent_nums: torch.Tensor, clusters: torch.Tensor, cluster_nums: torch.Tensor, k: int = 4):
        sent_mem = self._encode_sentences(article_sents, sent_nums)
        entity_mem = self._encode_entities(clusters, cluster_nums)
        B = article_sents.size(0)
        ptr_in = (sent_mem.mean(dim=1, keepdim=True) + entity_mem.mean(dim=1, keepdim=True)) / 2
        sent_nums_list = sent_nums.cpu().tolist() if isinstance(sent_nums, torch.Tensor) else sent_nums
        cluster_nums_list = cluster_nums.cpu().tolist() if isinstance(cluster_nums, torch.Tensor) else cluster_nums
        scores = self.pointer_decoder(sent_mem=sent_mem, entity_mem=entity_mem, ptr_in=ptr_in, sent_nums=sent_nums_list, entity_nums=cluster_nums_list)
        scores = scores.squeeze(1)
        mask = len_mask(sent_nums_list, scores.device)
        scores = scores.masked_fill(~mask, -1e9)
        probs = F.softmax(scores, dim=-1)
        _, topk = torch.topk(probs, k, dim=-1)
        return [idx.cpu().tolist() for idx in topk]

    @torch.no_grad()
    def extract_autoregressive(self, article_sents: torch.Tensor, sent_nums: torch.Tensor, clusters: torch.Tensor, cluster_nums: torch.Tensor, k: int = 4, max_extract: int = 10):
        sent_mem = self._encode_sentences(article_sents, sent_nums)
        entity_mem = self._encode_entities(clusters, cluster_nums)
        B = article_sents.size(0)
        sent_nums_list = sent_nums.cpu().tolist() if isinstance(sent_nums, torch.Tensor) else sent_nums
        cluster_nums_list = cluster_nums.cpu().tolist() if isinstance(cluster_nums, torch.Tensor) else cluster_nums
        extracted = [[] for _ in range(B)]
        ptr_in = (sent_mem.mean(dim=1, keepdim=True) + entity_mem.mean(dim=1, keepdim=True)) / 2
        for step in range(max_extract):
            scores = self.pointer_decoder(sent_mem=sent_mem, entity_mem=entity_mem, ptr_in=ptr_in, sent_nums=sent_nums_list, entity_nums=cluster_nums_list)
            scores = scores.squeeze(1)
            for b in range(B):
                for idx in extracted[b]:
                    scores[b, idx] = -1e9
                for idx in range(sent_nums_list[b], scores.size(1)):
                    scores[b, idx] = -1e9
            probs = F.softmax(scores, dim=-1)
            selected = torch.argmax(probs, dim=-1)
            for b in range(B):
                idx = selected[b].item()
                if idx < sent_nums_list[b] and idx not in extracted[b]:
                    extracted[b].append(idx)
            ptr_in = torch.gather(sent_mem, dim=1, index=selected.unsqueeze(-1).unsqueeze(-1).expand(B, 1, sent_mem.size(-1)))
            done = [len(ext) >= min(k, sent_nums_list[b]) for b, ext in enumerate(extracted)]
            if all(done):
                break
        return extracted

class AbstractGenerator(nn.Module):
    """
    Abstractive generator: BiLSTM encoder + CopyLSTMDecoder (with attention & copy).
    """

    def __init__(self, vocab_size: int, emb_dim: int, n_hidden: int, bidirectional: bool, n_layer: int, dropout: float = 0.1):
        super().__init__()
        self._vocab_size = vocab_size
        self._n_hidden = n_hidden
        self._n_layer = n_layer
        self._bidirectional = bidirectional
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._enc_lstm = nn.LSTM(input_size=emb_dim, hidden_size=n_hidden, num_layers=n_layer, bidirectional=bidirectional, batch_first=True, dropout=dropout if n_layer > 1 else 0.0)
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._dec_lstm_cells = MultiLayerLSTMCells(input_size=emb_dim * 2, hidden_size=n_hidden, num_layers=n_layer, dropout=dropout)
        self._dec_h = nn.Linear(enc_out_dim, n_hidden, bias=False)
        self._dec_c = nn.Linear(enc_out_dim, n_hidden, bias=False)
        self._attn_wm = nn.Parameter(torch.Tensor(enc_out_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        nn.init.xavier_normal_(self._attn_wm)
        nn.init.xavier_normal_(self._attn_wq)
        self._projection = nn.Sequential(nn.Linear(2 * n_hidden, n_hidden), nn.Tanh(), nn.Linear(n_hidden, emb_dim, bias=False))
        self._copy = _CopyLinear(context_dim=n_hidden, state_dim=n_hidden, input_dim=emb_dim)
        self._decoder = CopyLSTMDecoder(copy_linear=self._copy, embedding=self._embedding, lstm_cells=self._dec_lstm_cells, attn_w=self._attn_wq, projection=self._projection)

    def encode(self, article: torch.Tensor, art_lens: Optional[List[int]] = None):
        # article: [B, L] token ids
        enc_art, (h, c) = lstm_encoder(sequence=article, lstm=self._enc_lstm, seq_lens=art_lens, embedding=self._embedding)
        if self._bidirectional:
            # h,c: [num_layers*2, B, hidden]
            h = h.view(self._n_layer, 2, h.size(1), h.size(2))
            c = c.view(self._n_layer, 2, c.size(1), c.size(2))
            h = torch.cat([h[:, 0], h[:, 1]], dim=2)  # [n_layer, B, hidden*2]
            c = torch.cat([c[:, 0], c[:, 1]], dim=2)
        init_h = torch.stack([self._dec_h(s) for s in h], dim=0)  # [n_layer, B, n_hidden]
        init_c = torch.stack([self._dec_c(s) for s in c], dim=0)
        enc_proj = torch.matmul(enc_art, self._attn_wm)  # [B, L, n_hidden]
        mean_ctx = sequence_mean(enc_proj, art_lens, dim=1)
        prev_attn_out = self._projection(torch.cat([init_h[-1], mean_ctx], dim=1))
        init_states = ((init_h, init_c), prev_attn_out)
        return (enc_art, enc_proj), init_states

    def forward(self, article: torch.Tensor, art_lens: List[int], abstract: torch.Tensor, extend_art: torch.Tensor, extend_vsize: int):
        (enc_art, enc_proj), init_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, enc_art.device).unsqueeze(-2)
        decoder_input = (enc_art, enc_proj, mask, extend_art.to(enc_art.device), extend_vsize)
        logits = self._decoder(decoder_input, init_states, abstract) if hasattr(self._decoder, '__call__') else self._decode_teacher(decoder_input, init_states, abstract)
        # Note: earlier CopyLSTMDecoder._step returns log_prob; for teacher forcing you'd iterate and collect log_prob
        # If using a decoder.forward style, please adapt. For compatibility, we'll implement teacher for loop here:
        # But above we try direct call if implemented. If not, fallback:
        if not isinstance(logits, torch.Tensor):
            # implement teacher forcing manually
            B, T = abstract.size()
            device = article.device
            outputs = []
            states = init_states
            for t in range(T):
                tok = abstract[:, t:t+1].to(device)
                logp, states, _ = self._decoder._step(tok, states, decoder_input)
                outputs.append(logp)
            logits = torch.stack(outputs, dim=1)
        return logits  # [B, T, extend_vsize] (log probabilities)

    @torch.no_grad()
    def batched_beamsearch(self, article: torch.Tensor, art_lens: List[int], extend_art: torch.Tensor, extend_vsize: int,
                           go: int, eos: int, unk: int, max_len: int, beam_size: int, min_len: int = 1, diverse: float = 1.0):
        device = article.device
        batch_size = article.size(0)
        (enc_art, enc_proj), init_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, device).unsqueeze(-2)
        attention_input = (enc_art, enc_proj, mask, extend_art.to(device), extend_vsize)
        (h, c), prev_attn_out = init_states
        # initialize beams
        all_beams = [init_beam(go, (h[:, i, :], c[:, i, :], prev_attn_out[i])) for i in range(batch_size)]
        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]
        for t in range(max_len):
            toks = []
            all_states = []
            active_indices = []
            for i, beam in enumerate(all_beams):
                if not beam:
                    continue
                active_indices.append(i)
                token, states = pack_beam(beam, device)
                toks.append(token)
                all_states.append(states)
            if not toks:
                break
            token = torch.stack(toks, dim=1)  # [beam, num_active]
            # prepare states for decoder
            h_batch = torch.stack([s[0][0] for s in all_states], dim=2)  # [num_layers, num_active, hidden]
            c_batch = torch.stack([s[0][1] for s in all_states], dim=2)
            prev_out_batch = torch.stack([s[1] for s in all_states], dim=1)  # [num_active, emb_dim] -> adapt indexing in decoder
            # Ensure token values < vocab to avoid OOB
            token.masked_fill_(token >= self._vocab_size, unk)
            force_not_stop = t < min_len
            topk_idx, topk_logprob, new_states, attn_score = self._decoder.topk_step(token, ((h_batch, c_batch), prev_out_batch), attention_input, beam_size, force_not_stop=force_not_stop)
            batch_i = 0
            for i in active_indices:
                beam = all_beams[i]
                if not beam:
                    batch_i += 1
                    continue
                finished, new_beam = next_search_beam(beam=beam, beam_size=beam_size, finished=finished_beams[i], end_tok=eos,
                                                     topk_idx=topk_idx[:, batch_i, :], topk_logprob=topk_logprob[:, batch_i, :],
                                                     hists=(new_states[0][0][:, :, batch_i, :], new_states[0][1][:, :, batch_i, :], new_states[1][:, batch_i, :]),
                                                     attn=attn_score[:, batch_i, :] if attn_score is not None else None,
                                                     diverse=diverse)
                batch_i += 1
                if len(finished) >= beam_size:
                    all_beams[i] = []
                    outputs[i] = finished[:beam_size]
                else:
                    all_beams[i] = new_beam
                    finished_beams[i] = finished
            if all(outputs):
                break
        for i, (o, f, b) in enumerate(zip(outputs, finished_beams, all_beams)):
            if o is None:
                outputs[i] = (f + b)[:beam_size]
        return outputs



@META_ARCHITECTURE.register()
class SENECAModel(nn.Module):
    """
    Full SENECA integrated wrapper.
    Uses preprocessing modules from 'preprocessing' package if available.
    """
    def __init__(self, cfg, vocab: Vocab):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.d_model = getattr(cfg, "d_model", None)
        
        # Core SENECA components
        ext_cfg = cfg.SENECA.EXTRACTOR
        gen_cfg = cfg.SENECA.GENERATOR
        
        self.extractor = EntityAwareExtractor(
            vocab_size=self.vocab_size,
            emb_dim=ext_cfg.EMB_DIM,
            conv_hidden=ext_cfg.CONV_HIDDEN,
            lstm_hidden=ext_cfg.LSTM_HIDDEN,
            lstm_layer=ext_cfg.LSTM_LAYER,
            bidirectional=ext_cfg.BIDIRECTIONAL,
            n_hop=ext_cfg.N_HOP,
            dropout=ext_cfg.DROPOUT
        )
        
        self.generator = AbstractGenerator(
            vocab_size=self.vocab_size,
            emb_dim=gen_cfg.EMB_DIM,
            n_hidden=gen_cfg.N_HIDDEN,
            bidirectional=gen_cfg.BIDIRECTIONAL,
            n_layer=gen_cfg.N_LAYER,
            dropout=gen_cfg.DROPOUT
        )
        
        # Loss functions
        self.extraction_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.generation_loss = nn.CrossEntropyLoss(ignore_index=0)
        
        print(f"[SENECAAdapted] Initialized with vocab size: {self.vocab_size}")
    
    
    def _convert_to_seneca_format(self, input_ids: torch.Tensor):
        """
        Convert simple input_ids to SENECA format.
        
        Input:
            input_ids: [B, L] flat token sequence
        
        Output:
            article_sents: [B, max_sents, max_words]
            sent_nums: [B]
            clusters: [B, max_clusters, max_mentions] (dummy)
            cluster_nums: [B]
            extend_art: [B, L]
            extend_vsize: int
        """
        B, L = input_ids.size()
        device = input_ids.device
        
        # ============================================
        # 1. Split into sentences (simple heuristic)
        # ============================================
        # Strategy: split at <sep> token or every N tokens
        sep_token = getattr(self.vocab, 'sep_idx', getattr(self.vocab, 'eos_idx', 3))
        
        article_sents_list = []
        sent_nums = []
        
        for b in range(B):
            seq = input_ids[b]
            
            # Find sentence boundaries (at sep_token or every 20 tokens)
            sentences = []
            current_sent = []
            
            for token in seq:
                if token.item() == 0:  # padding
                    break
                
                current_sent.append(token.item())
                
                # End sentence at sep_token or max length
                if token.item() == sep_token or len(current_sent) >= 20:
                    if current_sent:
                        sentences.append(current_sent)
                        current_sent = []
            
            # Add remaining
            if current_sent:
                sentences.append(current_sent)
            
            if not sentences:  # fallback
                sentences = [seq[:20].cpu().tolist()]
            
            article_sents_list.append(sentences)
            sent_nums.append(len(sentences))
        
        # Pad sentences to same shape
        max_sents = max(sent_nums)
        max_words = max(max(len(s) for s in sents) for sents in article_sents_list)
        
        article_sents = torch.zeros(B, max_sents, max_words, dtype=torch.long, device=device)
        for i, sents in enumerate(article_sents_list):
            for j, sent in enumerate(sents):
                article_sents[i, j, :len(sent)] = torch.tensor(sent, device=device)
        
        # ============================================
        # 2. Create dummy entities (simplified)
        # ============================================
        # Strategy: extract frequent tokens as "pseudo-entities"
        max_clusters = 3
        max_mentions = 5
        clusters = torch.zeros(B, max_clusters, max_mentions, dtype=torch.long, device=device)
        cluster_nums = torch.ones(B, dtype=torch.long) * max_clusters
        
        for i in range(B):
            # Get top-3 most frequent tokens (excluding special tokens)
            seq = input_ids[i]
            valid_tokens = seq[seq > 3]  # exclude pad/unk/bos/eos
            
            if valid_tokens.numel() > 0:
                unique, counts = torch.unique(valid_tokens, return_counts=True)
                topk_values, topk_indices = torch.topk(counts, min(3, len(unique)))
                top_tokens = unique[topk_indices]
                
                for j, token in enumerate(top_tokens):
                    # Create pseudo-cluster: just repeat token 3 times
                    clusters[i, j, :3] = token
            else:
                # Fallback: use first non-special token
                clusters[i, 0, 0] = seq[seq > 3][0] if (seq > 3).any() else 4
        
        # ============================================
        # 3. Extended vocab (simplified - just copy input)
        # ============================================
        extend_art = input_ids.clone()
        extend_vsize = self.vocab_size  # no OOV for simplicity
        
        return {
            'article_sents': article_sents,
            'sent_nums': sent_nums,
            'clusters': clusters,
            'cluster_nums': cluster_nums,
            'extend_art': extend_art,
            'extend_vsize': extend_vsize
        }
    
    # ============================================
    # Training Forward
    # ============================================
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, 
                extend_art: Optional[torch.Tensor] = None,
                extend_vsize: Optional[int] = None):
        """
        Forward for training (compatible with TextSumTask).
        
        Args:
            input_ids: [B, L] source sequence
            labels: [B, T] target sequence
            extend_art: optional, default to input_ids
            extend_vsize: optional, default to vocab_size
        
        Returns:
            logits: [B, T, vocab_size]
            loss: scalar
        """
        B = input_ids.size(0)
        device = input_ids.device
        
        # Convert to SENECA format
        seneca_batch = self._convert_to_seneca_format(input_ids)
        
        # Use provided extend_art or default
        if extend_art is None:
            extend_art = seneca_batch['extend_art']
        if extend_vsize is None:
            extend_vsize = seneca_batch['extend_vsize']
        
        # ============================================
        # 1. Extraction (with pseudo-labels)
        # ============================================
        # Create pseudo extraction labels: select first 3 sentences
        max_extract = 3
        target_indices = torch.zeros(B, max_extract, dtype=torch.long, device=device)
        for i in range(B):
            num_sents = seneca_batch['sent_nums'][i]
            target_indices[i, :min(max_extract, num_sents)] = torch.arange(
                min(max_extract, num_sents), device=device
            )
        
        extraction_scores = self.extractor(
            article_sents=seneca_batch['article_sents'],
            sent_nums=torch.tensor(seneca_batch['sent_nums'], device=device),
            clusters=seneca_batch['clusters'],
            cluster_nums=seneca_batch['cluster_nums'],
            target_indices=target_indices
        )
        
        # Extraction loss (on pseudo-labels)
        ext_loss = self.extraction_loss(
            extraction_scores.view(-1, extraction_scores.size(-1)),
            target_indices.view(-1)
        )
        
        # ============================================
        # 2. Generation (on full input_ids)
        # ============================================
        # For training, use full input_ids as "extracted" content
        # This is simplified - in real SENECA, use actual extracted sentences
        
        art_lens = [input_ids[i].ne(0).sum().item() for i in range(B)]
        
        summary_logits = self.generator(
            article=input_ids,
            art_lens=art_lens,
            abstract=labels,
            extend_art=extend_art,
            extend_vsize=extend_vsize
        )
        
        # Generation loss
        gen_loss = self.generation_loss(
            summary_logits.view(-1, summary_logits.size(-1)),
            labels.view(-1)
        )
        
        # Combined loss
        total_loss = 0.3 * ext_loss + 0.7 * gen_loss
        
        return summary_logits, total_loss
    
    # ============================================
    # Inference (Prediction)
    # ============================================
    
    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor,
                extend_art: Optional[torch.Tensor] = None,
                extend_vsize: Optional[int] = None,
                max_len: int = 120,
                beam_size: int = 4,
                min_len: int = 10) -> List[List[int]]:
        """
        Predict for inference (compatible with TextSumTask).
        
        Args:
            input_ids: [B, L]
            extend_art: optional
            extend_vsize: optional
            max_len: max summary length
            beam_size: beam size
            min_len: min summary length
        
        Returns:
            List[List[int]] - generated token sequences
        """
        self.eval()
        B = input_ids.size(0)
        device = input_ids.device
        
        # Convert to SENECA format
        seneca_batch = self._convert_to_seneca_format(input_ids)
        
        if extend_art is None:
            extend_art = seneca_batch['extend_art']
        if extend_vsize is None:
            extend_vsize = seneca_batch['extend_vsize']
        
        # ============================================
        # 1. Extract sentences
        # ============================================
        k_extract = getattr(self.cfg.INFERENCE, 'TOP_K', 4)
        
        extracted_indices = self.extractor.extract(
            article_sents=seneca_batch['article_sents'],
            sent_nums=torch.tensor(seneca_batch['sent_nums'], device=device),
            clusters=seneca_batch['clusters'],
            cluster_nums=seneca_batch['cluster_nums'],
            k=k_extract
        )
        
        # Gather extracted sentences
        extracted_sents = self._get_sents_from_indices(
            seneca_batch['article_sents'], extracted_indices
        )
        
        art_lens = [
            extracted_sents[i].ne(0).sum().item() 
            for i in range(extracted_sents.size(0))
        ]
        
        # ============================================
        # 2. Generate summary
        # ============================================
        summaries = self.generator.batched_beamsearch(
            article=extracted_sents,
            art_lens=art_lens,
            extend_art=extend_art,
            extend_vsize=extend_vsize,
            go=getattr(self.vocab, 'bos_idx', 2),
            eos=getattr(self.vocab, 'eos_idx', 3),
            unk=getattr(self.vocab, 'unk_idx', 1),
            max_len=max_len,
            beam_size=beam_size,
            min_len=min_len,
            diverse=1.0
        )
        
        # Extract best sequences
        outputs = []
        for beam in summaries:
            if beam:
                best = beam[0]
                seq = best.sequence[1:] if best.sequence else []  # remove BOS
                outputs.append(seq)
            else:
                outputs.append([])
        
        return outputs
    
    # ============================================
    # Helper: gather sentences by indices
    # ============================================
    
    def _get_sents_from_indices(self, article_sents: torch.Tensor, 
                                 indices: List[List[int]]) -> torch.Tensor:
        """
        Gather sentences by indices.
        
        Args:
            article_sents: [B, max_sents, max_words]
            indices: List[List[int]] - sentence indices per sample
        
        Returns:
            [B, max_selected_words] flattened selected sentences
        """
        B = article_sents.size(0)
        device = article_sents.device
        
        batch_extracted = []
        
        for i in range(B):
            idx_list = indices[i] if i < len(indices) else [0]
            
            # Clamp to valid range
            idx_list = [
                idx for idx in idx_list 
                if 0 <= idx < article_sents.size(1)
            ]
            
            if not idx_list:
                idx_list = [0]
            
            # Gather sentences and flatten
            selected_sents = [article_sents[i, j] for j in sorted(set(idx_list))]
            
            # Concatenate sentences
            flattened = torch.cat(selected_sents, dim=0)
            
            # Remove padding
            flattened = flattened[flattened != 0]
            
            batch_extracted.append(flattened)
        
        # Pad to same length
        max_len = max(s.size(0) for s in batch_extracted)
        padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
        
        for i, seq in enumerate(batch_extracted):
            padded[i, :seq.size(0)] = seq
        
        return padded


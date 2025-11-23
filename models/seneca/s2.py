import math
import random
from typing import List, Optional, Tuple, Dict, Any
import torch
from torch import nn
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab
from collections import Counter

INI = 1e-2

# ---------------------------
# Utilities
# ---------------------------
def len_mask(lens: List[int], device: torch.device):
    """Return mask [B, max_len] True for valid positions"""
    if isinstance(lens, torch.Tensor):
        lens = lens.tolist()
    if not isinstance(lens, (list, tuple)) or len(lens) == 0:
        return torch.zeros((0, 0), dtype=torch.bool, device=device)
    max_len = max(lens)
    batch = len(lens)
    arange = torch.arange(max_len, device=device).expand(batch, max_len)
    lens_t = torch.tensor(lens, device=device).unsqueeze(1)
    return arange < lens_t

def sequence_mean(sequence: torch.Tensor, seq_lens: Optional[List[int]] = None, dim: int = 1):
    """Mean pooling over valid positions using seq_lens."""
    if seq_lens is None:
        return sequence.mean(dim=dim)
    mask = len_mask(seq_lens, sequence.device).unsqueeze(-1).float()
    summed = (sequence * mask).sum(dim=dim)
    lens_tensor = torch.tensor(seq_lens, device=sequence.device).unsqueeze(-1).float().clamp(min=1.0)
    return summed / lens_tensor

# LCS / ROUGE-L (used for greedy oracle sentence selection)
def lcs_length(a: List[int], b: List[int]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def rouge_l_f1(pred: List[int], ref: List[int]) -> float:
    if len(pred) == 0 or len(ref) == 0:
        return 0.0
    lcs = lcs_length(pred, ref)
    prec = lcs / max(1, len(pred))
    rec = lcs / max(1, len(ref))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

# ---------------------------
# Reusable building blocks
# ---------------------------
class ConvEncoder(nn.Module):
    """Multi-kernel 1D conv encoder over token embeddings."""
    def __init__(self, emb_dim: int, n_hidden: int, kernel_sizes: List[int], dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim, out_channels=n_hidden, kernel_size=k, padding=(k - 1) // 2)
            for k in kernel_sizes
        ])
        self.dropout = dropout

    def forward(self, emb_input: torch.Tensor) -> torch.Tensor:
        # emb_input: [B, L, D_emb]
        x = emb_input.transpose(1, 2)  # [B, D_emb, L]
        x = F.dropout(x, p=self.dropout, training=self.training)
        outs = [F.relu(conv(x)) for conv in self.convs]
        min_len = min(o.size(2) for o in outs)
        outs = [o[:, :, :min_len] for o in outs]
        cat = torch.cat(outs, dim=1)  # [B, n_hidden * n_kernels, L']
        return cat.transpose(1, 2)  # [B, L', n_hidden * n_kernels]

class MultiLayerLSTMCells(nn.Module):
    """Stacked LSTMCells for step-by-step decoding (supports teacher forcing)."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            insz = input_size if i == 0 else hidden_size
            self.cells.append(nn.LSTMCell(insz, hidden_size))
        self.dropout = dropout

    def forward(self, input_: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]):
        """
        input_: [B, input_size]
        states: (h_prev, c_prev) each [num_layers, B, hidden_size]
        returns: (layer_output [B, hidden], (new_h, new_c))
        """
        h_prev, c_prev = states
        new_h, new_c = [], []
        layer_output = None
        for i, cell in enumerate(self.cells):
            h_i = h_prev[i]
            c_i = c_prev[i]
            inp = input_ if i == 0 else layer_output
            h_i_new, c_i_new = cell(inp, (h_i, c_i))
            layer_output = h_i_new
            if self.training and self.dropout > 0 and i < self.num_layers - 1:
                layer_output = F.dropout(layer_output, p=self.dropout)
            new_h.append(h_i_new)
            new_c.append(c_i_new)
        new_h = torch.stack(new_h, dim=0)
        new_c = torch.stack(new_c, dim=0)
        return layer_output, (new_h, new_c)

class _CopyLinear(nn.Module):
    """Copy gate: v_c^T c + v_s^T s + v_i^T x + b"""
    def __init__(self, context_dim: int, state_dim: int, input_dim: int, bias: bool = True):
        super().__init__()
        self._v_c = nn.Parameter(torch.empty(context_dim))
        self._v_s = nn.Parameter(torch.empty(state_dim))
        self._v_i = nn.Parameter(torch.empty(input_dim))
        nn.init.uniform_(self._v_c, -INI, INI)
        nn.init.uniform_(self._v_s, -INI, INI)
        nn.init.uniform_(self._v_i, -INI, INI)
        self._b = nn.Parameter(torch.zeros(1)) if bias else None

    def forward(self, context: torch.Tensor, state: torch.Tensor, input_: torch.Tensor):
        out = (context @ self._v_c.unsqueeze(-1) +
               state @ self._v_s.unsqueeze(-1) +
               input_ @ self._v_i.unsqueeze(-1))
        if self._b is not None:
            out = out + self._b
        return out  # [B, 1]

def step_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mem_mask: Optional[torch.Tensor] = None):
    """
    query: [B, Dq]
    key: [B, L, Dk]
    value: [B, L, Dv]
    mem_mask: [B, L] (True for valid)
    returns: (context [B, Dv], attn_scores [B, L])
    """
    score = torch.bmm(query.unsqueeze(1), key.transpose(1, 2))  # [B,1,L]
    if mem_mask is not None:
        m = mem_mask.unsqueeze(1) if mem_mask.dim() == 2 else mem_mask
        score = score.masked_fill(~m, -1e9)
        norm = F.softmax(score, dim=-1)
    else:
        norm = F.softmax(score, dim=-1)
    ctx = torch.bmm(norm, value).squeeze(1)
    return ctx, norm.squeeze(1)

# ---------------------------
# Pointer net with entity attention
# ---------------------------
class LSTMPointerNet(nn.Module):
    def __init__(self, sent_dim: int, hidden: int, n_layer: int, side_dim: int):
        super().__init__()
        self._lstm = nn.LSTM(sent_dim, hidden, num_layers=n_layer, batch_first=True, dropout=0.0)
        # attentions
        self._attn_wm = nn.Parameter(torch.Tensor(sent_dim, hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(hidden, hidden))
        self._attn_v = nn.Parameter(torch.Tensor(hidden))
        # side/entity attention
        self.side_wm = nn.Parameter(torch.Tensor(side_dim, hidden))
        self.side_wq = nn.Parameter(torch.Tensor(hidden, hidden))
        self.side_v = nn.Parameter(torch.Tensor(hidden))
        self._attn_ws = nn.Parameter(torch.Tensor(hidden, hidden))
        # initialize
        for p in [self._attn_wm, self._attn_wq, self.side_wm, self.side_wq, self._attn_ws]:
            nn.init.xavier_normal_(p)
        for p in [self._attn_v, self.side_v]:
            nn.init.uniform_(p, -INI, INI)

    def forward(self, sent_mem: torch.Tensor, entity_mem: torch.Tensor, ptr_in: torch.Tensor, sent_nums: List[int], entity_nums: List[int]):
        """
        sent_mem: [B, Ns, D_sent]
        entity_mem: [B, Ne, D_ent]
        ptr_in: [B, 1, D_sent]
        returns: score [B, 1, Ns]
        """
        query, _ = self._lstm(ptr_in)  # [B, 1, Hq]
        query = query.squeeze(1)  # [B, Hq]
        # entity attention
        side_feat = entity_mem @ self.side_wm  # [B, Ne, H]
        entity_ctx = self._attention(side_feat, query, self.side_v, self.side_wq, entity_nums)  # [B, H]
        # sentence attention with entity side
        sent_feat = sent_mem @ self._attn_wm  # [B, Ns, H]
        score = self._attn_with_side(sent_feat, query, entity_ctx, self._attn_v, self._attn_wq, self._attn_ws, sent_nums)  # [B, Ns]
        return score.unsqueeze(1)

    @staticmethod
    def _attention(mem: torch.Tensor, query: torch.Tensor, v: torch.Tensor, w: torch.Tensor, sizes: List[int]):
        # mem: [B, L, H], query: [B, Hq]
        score = torch.tanh(mem.unsqueeze(1) + query.matmul(w).unsqueeze(2))
        score = score.matmul(v.unsqueeze(-1)).squeeze(-1)  # [B, L]
        mask = len_mask(sizes, score.device)
        score = score.masked_fill(~mask, -1e9)
        attn = F.softmax(score, dim=-1)
        return attn.matmul(mem)  # [B, H]

    @staticmethod
    def _attn_with_side(mem, query, ctx, v, wq, ws, sizes):
        # mem: [B, L, H], query: [B, H], ctx: [B, H]
        s = query.matmul(wq).unsqueeze(2)
        e = ctx.matmul(ws).unsqueeze(2)
        score = torch.tanh(mem.unsqueeze(1) + s + e).matmul(v.unsqueeze(-1)).squeeze(-1)
        mask = len_mask(sizes, score.device)
        score = score.masked_fill(~mask, -1e9)
        return score  # [B, L]

# ---------------------------
# Entity-aware content extractor
# ---------------------------
class EntityAwareExtractor(nn.Module):
    """
    Input:
      article_sents: [B, Ns, Nw] token ids
      sent_nums: List[int]
      clusters: [B, Ne, Nm] token ids for mentions
      cluster_nums: List[int]
    Output (training):
      scores: [B, T, Ns] (T = target steps)
    Output (inference):
      extracted indices list per batch
    """
    def __init__(self, vocab_size: int, emb_dim: int = 128, conv_hidden: int = 100,
                 sent_kernels: List[int] = [3,4,5], ent_kernels: List[int] = [2,3,4],
                 lstm_hidden: int = 256, lstm_layer: int = 1, bidirectional: bool = True, dropout: float = 0.1):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # sentence encoder
        self.sent_conv = ConvEncoder(emb_dim=emb_dim, n_hidden=conv_hidden, kernel_sizes=sent_kernels, dropout=dropout)
        self.sent_lstm = nn.LSTM(input_size=conv_hidden * len(sent_kernels), hidden_size=lstm_hidden, num_layers=lstm_layer, batch_first=True, bidirectional=bidirectional)
        # entity encoder
        self.ent_conv = ConvEncoder(emb_dim=emb_dim, n_hidden=conv_hidden, kernel_sizes=ent_kernels, dropout=dropout)
        ent_dim = conv_hidden * len(ent_kernels)
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.entity_proj = nn.Linear(ent_dim, enc_out_dim)
        # pointer
        self.pointer = LSTMPointerNet(sent_dim=enc_out_dim, hidden=enc_out_dim, n_layer=lstm_layer, side_dim=enc_out_dim)

    def _encode_sentences(self, article_sents: torch.Tensor, sent_nums: List[int]):
        B, Ns, Nw = article_sents.size()
        sents = self._embedding(article_sents)  # [B, Ns, Nw, emb]
        sents_flat = sents.view(B * Ns, Nw, -1)
        conv_sents = self.sent_conv(sents_flat)  # [B*Ns, L', H]
        sent_vecs = conv_sents.max(dim=1).values  # [B*Ns, H]
        sent_vecs = sent_vecs.view(B, Ns, -1)
        # pack by sent_nums (number of sentences per sample)
        if isinstance(sent_nums, torch.Tensor):
            sent_nums_list = sent_nums.cpu().tolist()
        else:
            sent_nums_list = sent_nums
        packed = nn.utils.rnn.pack_padded_sequence(sent_vecs, sent_nums_list, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.sent_lstm(packed)
        sent_mem, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return sent_mem  # [B, Ns, enc_out_dim]

    def _encode_entities(self, clusters: torch.Tensor, cluster_nums: List[int]):
        B, Ne, Nm = clusters.size()
        ent_emb = self._embedding(clusters)
        ent_flat = ent_emb.view(B * Ne, Nm, -1)
        conv_ents = self.ent_conv(ent_flat)
        ent_vecs = conv_ents.max(dim=1).values  # [B*Ne, ent_dim]
        ent_vecs = ent_vecs.view(B, Ne, -1)
        return ent_vecs

    def forward(self, article_sents: torch.Tensor, sent_nums: List[int],
                clusters: torch.Tensor, cluster_nums: List[int],
                target_indices: Optional[torch.Tensor] = None, max_extract: int = 4):
        sent_mem = self._encode_sentences(article_sents, sent_nums)  # [B, Ns, D]
        ent_mem = self._encode_entities(clusters, cluster_nums)      # [B, Ne, ent_dim]
        ent_proj = self.entity_proj(ent_mem)                         # [B, Ne, D]

        B, Ns, D = sent_mem.size()
        # initial pointer input: mean of sentence memory + entity proj mean
        ptr_in = (sent_mem.mean(dim=1, keepdim=True) + ent_proj.mean(dim=1, keepdim=True)) * 0.5  # [B,1,D]

        if target_indices is not None:
            T = target_indices.size(1)
            scores_all = []
            for t in range(T):
                score = self.pointer(sent_mem, ent_proj, ptr_in, sent_nums, cluster_nums)  # [B,1,Ns]
                scores_all.append(score.squeeze(1))
                # teacher forcing: update ptr_in with gold
                idx = target_indices[:, t].clamp(0, Ns - 1)
                ptr_in = torch.gather(sent_mem, dim=1, index=idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, D))
            scores = torch.stack(scores_all, dim=1)  # [B, T, Ns]
            return scores
        else:
            # greedy autoregressive extraction
            extracted = [[] for _ in range(B)]
            for step in range(max_extract):
                score = self.pointer(sent_mem, ent_proj, ptr_in, sent_nums, cluster_nums).squeeze(1)  # [B, Ns]
                mask = len_mask(sent_nums, score.device)
                score = score.masked_fill(~mask, -1e9)
                probs = F.softmax(score, dim=-1)
                sel = probs.argmax(dim=-1)  # [B]
                for b in range(B):
                    idx = int(sel[b].item())
                    if idx < sent_nums[b] and idx not in extracted[b]:
                        extracted[b].append(idx)
                ptr_in = torch.gather(sent_mem, dim=1, index=sel.unsqueeze(-1).unsqueeze(-1).expand(B, 1, D))
            return extracted

    @torch.no_grad()
    def extract_topk_nonautoregressive(self, article_sents: torch.Tensor, sent_nums: List[int],
                                       clusters: torch.Tensor, cluster_nums: List[int], k: int = 4):
        """Return top-k sentences by mean scoring (non-autoregressive)."""
        sent_mem = self._encode_sentences(article_sents, sent_nums)
        ent_mem = self._encode_entities(clusters, cluster_nums)
        ent_proj = self.entity_proj(ent_mem)
        ptr_in = (sent_mem.mean(dim=1, keepdim=True) + ent_proj.mean(dim=1, keepdim=True)) * 0.5
        scores = self.pointer(sent_mem, ent_proj, ptr_in, sent_nums, cluster_nums).squeeze(1)  # [B, Ns]
        mask = len_mask(sent_nums, scores.device)
        scores = scores.masked_fill(~mask, -1e9)
        probs = F.softmax(scores, dim=-1)
        _, topk = torch.topk(probs, k=min(k, scores.size(1)), dim=-1)
        return [list(topk[i, :min(k, sent_nums[i])].cpu().tolist()) for i in range(topk.size(0))]

# ---------------------------
# Abstractive generator (Pointer-generator)
# ---------------------------
class CopyLSTMDecoder(nn.Module):
    """Decoder wrapper that supports teacher forcing and greedy decode."""
    def __init__(self, embedding: nn.Embedding, lstm_cells: MultiLayerLSTMCells, attn_w: nn.Linear, projection: nn.Module, copy_linear: _CopyLinear):
        super().__init__()
        self._embedding = embedding
        self._lstm_cells = lstm_cells
        self._attn_w = attn_w
        self._projection = projection
        self._copy = copy_linear

    def _step(self, tok: torch.Tensor, states: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], attention: Tuple):
        (prev_h, prev_c), prev_out = states
        enc_mem, enc_proj, mask, extend_art, extend_vsize = attention
        emb = self._embedding(tok).squeeze(1)
        if prev_out.dim() == 3:
            prev_out = prev_out.squeeze(1)
        lstm_in = torch.cat([emb, prev_out], dim=-1)
        lstm_out, (new_h, new_c) = self._lstm_cells(lstm_in, (prev_h, prev_c))
        query = lstm_out @ self._attn_w  # [B, D]
        ctx, score = step_attention(query, enc_proj, enc_mem, mask)  # ctx [B, Denc], score [B, L]
        dec_out = self._projection(torch.cat([lstm_out, ctx], dim=-1))  # [B, emb_dim]
        logits = dec_out @ self._embedding.weight.T  # [B, V]
        if extend_vsize > logits.size(1):
            pad = torch.full((logits.size(0), extend_vsize - logits.size(1)), -1e8, device=logits.device)
            logits = torch.cat([logits, pad], dim=1)
        gen_prob = F.softmax(logits, dim=-1)
        copy_gate = torch.sigmoid(self._copy(ctx, lstm_out, emb))  # [B,1]
        if copy_gate.dim() == 1:
            copy_gate = copy_gate.unsqueeze(-1)
        copy_vals = score * copy_gate  # [B, L]
        add_tensor = torch.zeros_like(gen_prob)
        if extend_art.dim() == 1:
            extend_art = extend_art.unsqueeze(0).expand(add_tensor.size(0), -1)
        # clamp indices
        extend_art_clamped = extend_art.clamp(max=extend_vsize - 1)
        add_tensor.scatter_add_(1, extend_art_clamped.long(), copy_vals)
        final_prob = (1 - copy_gate) * gen_prob + add_tensor
        log_prob = torch.log(final_prob + 1e-12)
        new_states = ((new_h, new_c), dec_out)
        return log_prob, new_states, score

    def forward(self, attention: Tuple, init_states: Tuple, abstract: torch.Tensor):
        B, T = abstract.size()
        states = init_states
        outputs = []
        for t in range(T):
            tok = abstract[:, t:t+1]
            logp, states, _ = self._step(tok, states, attention)
            outputs.append(logp)
        return torch.stack(outputs, dim=1)  # [B, T, V_ext]

    def greedy_decode(self, attention: Tuple, init_states: Tuple, go: int, eos: int, max_len: int, min_len: int = 1):
        (h, c), prev_out = init_states
        B = h.size(1)
        device = h.device
        current_token = torch.full((B, 1), go, dtype=torch.long, device=device)
        generated = []
        logps = []
        states = init_states
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for t in range(max_len):
            log_prob, states, _ = self._step(current_token, states, attention)
            if t < min_len:
                log_prob[:, eos] = -1e20
            sel_logp, sel_tok = log_prob.max(dim=-1)
            generated.append(sel_tok)
            logps.append(sel_logp)
            finished = finished | (sel_tok == eos)
            if finished.all():
                break
            current_token = sel_tok.unsqueeze(1)
        generated = torch.stack(generated, dim=1)
        logps = torch.stack(logps, dim=1)
        return generated, logps

class AbstractGenerator(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, n_hidden: int = 256, bidirectional: bool = True, n_layer: int = 1, dropout: float = 0.1):
        super().__init__()
        self._vocab_size = vocab_size
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._enc_lstm = nn.LSTM(input_size=emb_dim, hidden_size=n_hidden, num_layers=n_layer, bidirectional=bidirectional, batch_first=True, dropout=dropout if n_layer>1 else 0.0)
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        # decoder cells: input = embedding + prev_attn_out (embedded dim + emb_dim)
        self._dec_cells = MultiLayerLSTMCells(input_size=emb_dim*2, hidden_size=enc_out_dim, num_layers=n_layer, dropout=dropout)
        self._dec_h = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        self._dec_c = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        self._attn_wm = nn.Parameter(torch.Tensor(enc_out_dim, enc_out_dim))
        self._attn_wq = nn.Parameter(torch.Tensor(enc_out_dim, enc_out_dim))
        nn.init.xavier_normal_(self._attn_wm)
        nn.init.xavier_normal_(self._attn_wq)
        self._projection = nn.Sequential(nn.Linear(2*enc_out_dim, enc_out_dim), nn.Tanh(), nn.Linear(enc_out_dim, emb_dim, bias=False))
        self._copy = _CopyLinear(context_dim=enc_out_dim, state_dim=enc_out_dim, input_dim=emb_dim)
        self._decoder = CopyLSTMDecoder(copy_linear=self._copy, embedding=self._embedding, lstm_cells=self._dec_cells, attn_w=self._attn_wq, projection=self._projection)

    def encode(self, article: torch.Tensor, art_lens: Optional[List[int]] = None):
        """
        article: [B, L] token ids
        art_lens: list of lengths
        """
        # pack + encode with embedding
        enc_art, (h, c) = lstm_encoder(article, self._enc_lstm, seq_lens=art_lens, embedding=self._embedding)
        # handle bidir
        if self._enc_lstm.bidirectional:
            # reshape and concat
            L = self._enc_lstm.num_layers
            h = h.view(L, 2, h.size(1), h.size(2))
            c = c.view(L, 2, c.size(1), c.size(2))
            h = torch.cat([h[:, 0], h[:, 1]], dim=2)
            c = torch.cat([c[:, 0], c[:, 1]], dim=2)
        # init decoder states
        init_h = torch.stack([self._dec_h(s) for s in h], dim=0)
        init_c = torch.stack([self._dec_c(s) for s in c], dim=0)
        enc_proj = torch.matmul(enc_art, self._attn_wm)
        mean_ctx = sequence_mean(enc_proj, art_lens, dim=1)
        prev_attn_out = self._projection(torch.cat([init_h[-1], mean_ctx], dim=1))
        init_states = ((init_h, init_c), prev_attn_out)
        return (enc_art, enc_proj), init_states

    def forward(self, article: torch.Tensor, art_lens: List[int], abstract: torch.Tensor, extend_art: torch.Tensor, extend_vsize: int):
        (enc_art, enc_proj), init_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, enc_art.device).unsqueeze(-2)
        attention = (enc_art, enc_proj, mask, extend_art.to(enc_art.device), extend_vsize)
        logits = self._decoder(attention, init_states, abstract)
        return logits

    @torch.no_grad()
    def greedy_generate(self, article: torch.Tensor, art_lens: List[int], extend_art: torch.Tensor, extend_vsize: int, go: int, eos: int, max_len: int = 120, min_len: int = 1):
        (enc_art, enc_proj), init_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, enc_art.device).unsqueeze(-2)
        attention = (enc_art, enc_proj, mask, extend_art.to(enc_art.device), extend_vsize)
        generated, log_probs = self._decoder.greedy_decode(attention, init_states, go, eos, max_len, min_len)
        # convert to python lists and strip after eos
        outputs = []
        for i in range(generated.size(0)):
            seq = generated[i].cpu().tolist()
            if eos in seq:
                seq = seq[:seq.index(eos)]
            outputs.append(seq)
        return outputs

# Helper: LSTM encoder (supports packing)
def lstm_encoder(sequence: torch.Tensor, lstm: nn.LSTM, seq_lens: Optional[List[int]] = None, init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, embedding: Optional[nn.Embedding] = None):
    if embedding is not None:
        sequence = embedding(sequence)
    if seq_lens is not None:
        if isinstance(seq_lens, torch.Tensor):
            seq_lens = seq_lens.cpu().tolist()
        sort_ind = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i], reverse=True)
        seq_sorted = sequence.index_select(0, torch.LongTensor(sort_ind).to(sequence.device))
        lens_sorted = [seq_lens[i] for i in sort_ind]
        packed = nn.utils.rnn.pack_padded_sequence(seq_sorted, lens_sorted, batch_first=True)
        packed_out, final_states = lstm(packed, init_states)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # restore original order
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder = [back_map[i] for i in range(len(sort_ind))]
        out = out.index_select(0, torch.LongTensor(reorder).to(out.device))
        h, c = final_states
        h = h.index_select(1, torch.LongTensor(reorder).to(h.device))
        c = c.index_select(1, torch.LongTensor(reorder).to(c.device))
        return out, (h, c)
    else:
        out, final_states = lstm(sequence, init_states)
        return out, final_states

# ---------------------------
# Extended vocab builder (copy generator)
# ---------------------------
def build_extended_for_batch(articles: torch.Tensor, vocab_size: int):
    """
    articles: [B, L] flattened articles (no pad tok allowed as actual PAD)
    returns: extend_art [B, L'], extend_vsize int
    """
    B, L = articles.size()
    device = articles.device
    extend_list = []
    extend_vsizes = []
    for b in range(B):
        nonpad = articles[b][articles[b] != 0].tolist()
        oov2idx = {}
        extended = []
        for tok in nonpad:
            if tok < vocab_size:
                extended.append(tok)
            else:
                if tok not in oov2idx:
                    oov2idx[tok] = len(oov2idx)
                extended.append(vocab_size + oov2idx[tok])
        extend_list.append(torch.tensor(extended, device=device) if len(extended) > 0 else torch.tensor([], device=device, dtype=torch.long))
        extend_vsizes.append(vocab_size + len(oov2idx))
    max_len = max((t.size(0) for t in extend_list), default=1)
    extend_art = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(extend_list):
        if seq.numel() > 0:
            extend_art[i, : seq.size(0)] = seq
    extend_vsize = max(extend_vsizes) if extend_vsizes else vocab_size
    return extend_art, extend_vsize

# ---------------------------
# Helper: convert input flat article -> seneca format (sentences + clusters)
# ---------------------------
def flatten_sentences_from_input(input_ids: torch.Tensor, vocab) -> Dict[str, Any]:
    """
    Input:
      input_ids: [B, L] flat tokens (with separator token in vocab or EOS fallback)
    Output:
      dict with keys:
        article_sents: [B, max_sents, max_words]
        sent_nums: List[int]
        clusters: [B, max_clusters, max_mentions] (currently simple heuristic zeros)
        cluster_nums: List[int]
        flat_articles: [B, max_flat_len]
    """
    B, L = input_ids.size()
    device = input_ids.device
    # choose sep token if available
    if hasattr(vocab, 'sep_idx'):
        sep_token = getattr(vocab, 'sep_idx')
    else:
        # if not present, fallback to eos
        sep_token = getattr(vocab, 'eos_idx', 2)
    article_sents_list = []
    sent_nums = []
    flat_articles_list = []
    for b in range(B):
        seq = input_ids[b]
        sentences = []
        cur = []
        for tok in seq:
            if int(tok.item()) == 0:
                break
            cur.append(int(tok.item()))
            if int(tok.item()) == sep_token:
                if cur:
                    sentences.append(cur.copy())
                    cur = []
        if cur:
            sentences.append(cur)
        if not sentences:
            # fallback: first L//4 tokens
            sentences = [seq[:max(1, L//4)].cpu().tolist()]
        article_sents_list.append(sentences)
        sent_nums.append(len(sentences))
        # flatten
        flat = []
        for s in sentences:
            flat.extend(s)
        if not flat:
            flat = [0]
        flat_articles_list.append(torch.tensor(flat, dtype=torch.long, device=device))
    max_sents = max(sent_nums)
    max_words = max(max(len(s) for s in sents) for sents in article_sents_list)
    article_sents = torch.zeros(B, max_sents, max_words, dtype=torch.long, device=device)
    for i, sents in enumerate(article_sents_list):
        for j, sent in enumerate(sents):
            article_sents[i, j, :len(sent)] = torch.tensor(sent, dtype=torch.long, device=device)
    # clusters: placeholder zeros; in production replace by CoreNLP clusters
    max_clusters = 4
    max_mentions = 6
    clusters = torch.zeros(B, max_clusters, max_mentions, dtype=torch.long, device=device)
    cluster_nums = [max_clusters] * B
    # flat_articles padded
    max_flat = max(x.size(0) for x in flat_articles_list)
    flat_articles = torch.zeros(B, max_flat, dtype=torch.long, device=device)
    for i, seq in enumerate(flat_articles_list):
        flat_articles[i, : seq.size(0)] = seq
    return {
        'article_sents': article_sents,
        'sent_nums': sent_nums,
        'clusters': clusters,
        'cluster_nums': cluster_nums,
        'flat_articles': flat_articles
    }

# ---------------------------
# Oracle greedy sentence selection (ROUGE-L)
# ---------------------------
def greedy_oracle_select(article_sents_list: List[List[int]], ref_tokens: List[int], k: int):
    """Greedy selection (per sample) maximizing ROUGE-L F1 cumulatively."""
    selected = []
    current_concat = []
    available = set(range(len(article_sents_list)))
    for _ in range(min(k, len(article_sents_list))):
        best_gain = 0.0
        best_idx = None
        best_concat = None
        for idx in available:
            cand = current_concat + article_sents_list[idx]
            gain = rouge_l_f1(cand, ref_tokens)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_concat = cand
        if best_idx is None:
            break
        selected.append(best_idx)
        current_concat = best_concat
        available.remove(best_idx)
    return selected

# ---------------------------
# Utils: gather selected sentences -> flattened tensor
# ---------------------------
def gather_selected_sentences(article_sents: torch.Tensor, indices: List[List[int]]) -> torch.Tensor:
    """article_sents: [B, max_sents, max_words]. indices: List[List[int]] -> padded [B, L_flat]"""
    B = article_sents.size(0)
    device = article_sents.device
    flattened = []
    for b in range(B):
        idxs = indices[b]
        toks = []
        for idx in idxs:
            if idx >= 0 and idx < article_sents.size(1):
                sent = article_sents[b, idx]
                nz = sent[sent != 0]
                toks.extend(nz.tolist())
        if not toks:
            toks = [0]
        flattened.append(torch.tensor(toks, dtype=torch.long, device=device))
    max_len = max([s.size(0) for s in flattened])
    out = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(flattened):
        out[i, : seq.size(0)] = seq
    return out

# ---------------------------
# Final registered SENECA model
# ---------------------------
@META_ARCHITECTURE.register()
class SENECAModel(nn.Module):
    """
    Full SENECA (Version B) model (no RL). Combines:
      - EntityAwareExtractor
      - AbstractGenerator (pointer-generator)
    API:
      __init__(cfg, vocab)
      forward(input_ids, labels=None, max_extract=3) -> when labels provided computes losses (ext+gen)
      predict(input_ids, k, max_len, min_len) -> generated summaries
    """
    def __init__(self, cfg, vocab: Vocab):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        # vocab size expected to be accessible via len(vocab) or vocab.vocab_size
        self.vocab_size = len(vocab) 
        ext_cfg = getattr(cfg, "SENECA", {}).get("EXTRACTOR", {}) if hasattr(cfg, "SENECA") else getattr(cfg, "EXTRACTOR", None)
        # fallback small defaults if config is not dict-like
        if ext_cfg is None or not hasattr(ext_cfg, 'EMB_DIM'):
            # minimal default object-ish
            class _C: pass
            ext_cfg = _C()
            ext_cfg.EMB_DIM = 128
            ext_cfg.CONV_HIDDEN = 100
            ext_cfg.LSTM_HIDDEN = 256
            ext_cfg.LSTM_LAYER = 1
            ext_cfg.BIDIRECTIONAL = True
            ext_cfg.N_HOP = 1
            ext_cfg.DROPOUT = 0.1
        gen_cfg = getattr(cfg, "SENECA", {}).get("GENERATOR", {}) if hasattr(cfg, "SENECA") else getattr(cfg, "GENERATOR", None)
        if gen_cfg is None or not hasattr(gen_cfg, 'EMB_DIM'):
            class _G: pass
            gen_cfg = _G()
            gen_cfg.EMB_DIM = 128
            gen_cfg.N_HIDDEN = 256
            gen_cfg.BIDIRECTIONAL = True
            gen_cfg.N_LAYER = 1
            gen_cfg.DROPOUT = 0.1

        # build modules
        self.extractor = EntityAwareExtractor(
            vocab_size=self.vocab_size,
            emb_dim=ext_cfg.EMB_DIM,
            conv_hidden=ext_cfg.CONV_HIDDEN,
            sent_kernels=[3,4,5],
            ent_kernels=[2,3,4],
            lstm_hidden=ext_cfg.LSTM_HIDDEN,
            lstm_layer=ext_cfg.LSTM_LAYER,
            bidirectional=ext_cfg.BIDIRECTIONAL,
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
        # losses (ignore pad idx = 0)
        self.extraction_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.generation_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def _convert_to_seneca_format(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """Convert flat input_ids -> article_sents, sent_nums, clusters, cluster_nums, flat_articles."""
        return flatten_sentences_from_input(input_ids, self.vocab)

    def _build_extended_for_batch(self, flat_articles: torch.Tensor):
        return build_extended_for_batch(flat_articles, vocab_size=self.vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, max_extract: int = 3):
        """
        If labels provided:
          - build oracle targets (greedy ROUGE-L)
          - compute extraction loss + generation loss
          - return (logits, total_loss, diagnostics)
        If labels not provided:
          - return extractor scores (for training extractor) or for inference call predict()
        """
        device = input_ids.device
        B = input_ids.size(0)

        seneca_batch = self._convert_to_seneca_format(input_ids)
        article_sents = seneca_batch['article_sents']          # [B, Ns, Nw]
        sent_nums = seneca_batch['sent_nums']                 # List[int]
        clusters = seneca_batch['clusters']
        cluster_nums = seneca_batch['cluster_nums']
        flat_articles = seneca_batch['flat_articles']

        # build extended mapping for entire article (not used by generator directly; generator uses extracted flat)
        extend_art_all, extend_vsize_all = self._build_extended_for_batch(flat_articles)

        # If user doesn't pass labels: return extractor-only outputs (scores) to train separately
        if labels is None:
            # return extractor scores with teacher forcing disabled (i.e., just extraction indices)
            extracted = self.extractor(article_sents, sent_nums, clusters, cluster_nums, target_indices=None, max_extract=max_extract)
            return extracted

        # --- Build oracle target indices per sample via greedy ROUGE-L ---
        # build list-of-sentences tokens per sample
        target_indices = torch.zeros(B, max_extract, dtype=torch.long, device=device)
        oracle_lists = []
        for b in range(B):
            num_s = sent_nums[b]
            sents = []
            for j in range(num_s):
                sent = article_sents[b, j]
                toks = sent[sent != 0].cpu().tolist()
                sents.append(toks)
            # reference tokens from labels
            ref = labels[b]
            ref_toks = ref[ref != 0].cpu().tolist()
            sel = greedy_oracle_select(sents, ref_toks, max_extract)
            oracle_lists.append(sel)
            for i, idx in enumerate(sel):
                target_indices[b, i] = idx

        # --- Extraction: teacher forcing with oracle targets ---
        extraction_scores = self.extractor(article_sents, sent_nums, clusters, cluster_nums, target_indices=target_indices)
        # extraction_scores: [B, T, Ns]
        B2, T, Ns = extraction_scores.size()
        ext_loss = self.extraction_loss_fn(extraction_scores.view(-1, Ns), target_indices.view(-1))

        # --- Build extracted sentences for generator input ---
        # Use mean over T to get per-sentence score then top-k (avoid flattening t*Ns)
        mean_scores = extraction_scores.mean(dim=1)  # [B, Ns]
        extracted_indices = []
        for b in range(B):
            valid = sent_nums[b]
            scores = mean_scores[b, :valid]
            if scores.numel() == 0:
                extracted_indices.append([0])
                continue
            k = min(max_extract, valid)
            _, topk = torch.topk(scores, k=k)
            extracted_indices.append([int(x.item()) for x in topk])

        extracted_sents = gather_selected_sentences(article_sents, extracted_indices)  # [B, L_ext]
        art_lens = [int((extracted_sents[i] != 0).sum().item()) for i in range(B)]
        extracted_extend_art, extracted_extend_vsize = self._build_extended_for_batch(extracted_sents)

        # --- Generation (teacher forcing) ---
        summary_logits = self.generator(article=extracted_sents, art_lens=art_lens, abstract=labels, extend_art=extracted_extend_art, extend_vsize=extracted_extend_vsize)
        gen_loss = self.generation_loss_fn(summary_logits.view(-1, summary_logits.size(-1)), labels.view(-1))

        # combined loss (weights can be tuned externally)
        total_loss = 0.3 * ext_loss + 0.7 * gen_loss

        diagnostics = {
            "ext_loss": ext_loss.item(),
            "gen_loss": gen_loss.item(),
            "total_loss": total_loss.item(),
            "extracted_indices": extracted_indices,
            "oracle_lists": oracle_lists
        }
        return summary_logits, total_loss, diagnostics

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, k: int = 4, max_len: int = 120, min_len: int = 10):
        """Greedy inference: extract k sentences then generate summary (greedy decoding)"""
        device = input_ids.device
        seneca_batch = self._convert_to_seneca_format(input_ids)
        article_sents = seneca_batch['article_sents']
        sent_nums = seneca_batch['sent_nums']
        clusters = seneca_batch['clusters']
        cluster_nums = seneca_batch['cluster_nums']

        # 1) extract
        extracted_indices = self.extractor.extract_greedy(article_sents=article_sents, sent_nums=torch.tensor(sent_nums, device=device),
                                                         clusters=clusters, cluster_nums=cluster_nums, k=k)
        extracted_sents = gather_selected_sentences(article_sents, extracted_indices)
        art_lens = [int((extracted_sents[i] != 0).sum().item()) for i in range(extracted_sents.size(0))]
        extracted_extend_art, extracted_extend_vsize = self._build_extended_for_batch(extracted_sents)

        # 2) generate
        summaries = self.generator.greedy_generate(article=extracted_sents, art_lens=art_lens, extend_art=extracted_extend_art,
                                                   extend_vsize=extracted_extend_vsize,
                                                   go=getattr(self.vocab, 'bos_idx', 1),
                                                   eos=getattr(self.vocab, 'eos_idx', 2),
                                                   max_len=max_len, min_len=min_len)
        return summaries

# End of file

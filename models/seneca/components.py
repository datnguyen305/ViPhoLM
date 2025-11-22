# models/seneca/components.py
import torch
from torch import nn
import torch.nn.init as init
from torch.nn import functional as F
from models.seneca.utils import len_mask, step_attention

INI = 1e-2
class ConvEncoder(nn.Module):
    """
    Multi-kernel CNN encoder for sequence encoding.
    """

    def __init__(self, emb_dim, n_hidden, kernel_sizes, dropout):
        super().__init__()
        self._convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=emb_dim,
                out_channels=n_hidden,
                kernel_size=k,
                padding=(k - 1) // 2  # better "same" padding
            )
            for k in kernel_sizes
        ])
        self._dropout = dropout
        self._n_kernels = len(kernel_sizes)
        self._n_hidden = n_hidden

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, emb_dim, seq_len]
        outs = []
        for conv in self._convs:
            K = conv.kernel_size[0]
            if x.size(2) < K:
                continue
            outs.append(F.relu(conv(x)))
        if not outs:
            smallest_conv = self._convs[0]
            outs.append(F.relu(smallest_conv(x)))

        outs = [out.max(dim=2).values for out in outs]  # [B, hidden] mỗi kernel
        out = torch.cat(outs, dim=1)                    # [B, 3*hidden]
        return out


class MultiLayerLSTMCells(nn.Module):
    """Unrolled multi-layer LSTM for flexible decoding."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.cells = nn.ModuleList([
            nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, input_, states):
        h_prev, c_prev = states
        h_next, c_next = [], []
        layer_input = input_
        for i, cell in enumerate(self.cells):
            h_i, c_i = cell(layer_input, (h_prev[i], c_prev[i]))
            h_next.append(h_i)
            c_next.append(c_i)
            if i < self.num_layers - 1:
                layer_input = F.dropout(h_i, p=self.dropout, training=self.training)
            else:
                layer_input = h_i
        return torch.stack(h_next), torch.stack(c_next)

class LSTMPointerNet(nn.Module):
    """
    Pointer Network Decoder with joint attention over sentences + entities.
    """

    def __init__(self, input_dim, n_hidden, n_layer, dropout, n_hop, side_dim):
        super().__init__()
        self._lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=n_hidden,
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0.0,
            batch_first=True
        )
        # Sentence attention
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        # Entity attention
        self.side_wm = nn.Parameter(torch.Tensor(side_dim, n_hidden))
        self.side_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.side_v = nn.Parameter(torch.Tensor(n_hidden))
        # Fusion
        self._attn_ws = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        # Init
        for p in [self._attn_wm, self._attn_wq, self.side_wm, self.side_wq, self._attn_ws]:
            init.xavier_normal_(p)
        for p in [self._attn_v, self.side_v]:
            init.uniform_(p, -INI, INI)

    def forward(self, sent_mem, entity_mem, ptr_in, sent_nums, entity_nums):
        query, _ = self._lstm(ptr_in)
        side_feat = entity_mem @ self.side_wm  # [B, Ne, H]
        entity_ctx = self._attention(side_feat, query, self.side_v, self.side_wq, entity_nums)
        sent_feat = sent_mem @ self._attn_wm
        score = self._attn_with_side(sent_feat, query, entity_ctx,
                                     self._attn_v, self._attn_wq, self._attn_ws,
                                     sent_nums)
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
        score.masked_fill_(~mask, -1e9)
        return score


class _CopyLinear(nn.Module):
    """Linear layer for copy gate computation."""

    def __init__(self, context_dim, state_dim, input_dim, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.empty(context_dim))
        self._v_s = nn.Parameter(torch.empty(state_dim))
        self._v_i = nn.Parameter(torch.empty(input_dim))
        init.uniform_(self._v_c, -INI, INI)
        init.uniform_(self._v_s, -INI, INI)
        init.uniform_(self._v_i, -INI, INI)
        self._b = nn.Parameter(torch.zeros(1)) if bias else None

    def forward(self, context, state, input_):
        out = (context @ self._v_c.unsqueeze(-1) +
               state @ self._v_s.unsqueeze(-1) +
               input_ @ self._v_i.unsqueeze(-1))
        if self._b is not None:
            out += self._b
        return out


class AttentionalLSTMDecoder(nn.Module):
    """Base decoder with attention."""

    def __init__(self, embedding, lstm_cells, attn_w, projection):
        super().__init__()
        self._embedding = embedding
        self._lstm = lstm_cells
        self._attn_w = attn_w
        self._projection = projection


class CopyLSTMDecoder(AttentionalLSTMDecoder):
    """LSTM Decoder with Copy Mechanism (See et al. 2017)."""

    def __init__(self, copy_linear, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._copy = copy_linear

    def _step(self, tok, states, attention):
        prev_states, prev_out = states
        emb = self._embedding(tok).squeeze(1)
        lstm_in = torch.cat([emb, prev_out], dim=-1)
        new_states = self._lstm(lstm_in, prev_states)
        lstm_out = new_states[0][-1]

        enc_mem, enc_proj, mask, extend_art, extend_vsize = attention
        query = lstm_out @ self._attn_w
        ctx, score = step_attention(query, enc_proj, enc_mem, mask)

        dec_out = self._projection(torch.cat([lstm_out, ctx], dim=-1))
        gen_prob = self._compute_gen_prob(dec_out, extend_vsize)
        copy_gate = torch.sigmoid(self._copy(ctx, lstm_out, emb))
        final_prob = (1 - copy_gate) * gen_prob
        final_prob = final_prob.scatter_add(1, extend_art, score * copy_gate)
        log_prob = torch.log(final_prob + 1e-8)
        return log_prob, (new_states, dec_out), score

    def _compute_gen_prob(self, dec_out, extend_vsize, eps=1e-6):
        logits = dec_out @ self._embedding.weight.T
        if extend_vsize > logits.size(1):
            pad = torch.full((logits.size(0), extend_vsize - logits.size(1)), eps, device=logits.device)
            logits = torch.cat([logits, pad], dim=1)
        return F.softmax(logits, dim=-1)

    def topk_step(self, tok, states, attention, beam_size, force_not_stop=False):
        log_prob, new_states, score = self._step(tok, states, attention)
        topk_logprob, topk_idx = torch.topk(log_prob, beam_size, dim=-1)
        return topk_idx, topk_logprob, new_states, score
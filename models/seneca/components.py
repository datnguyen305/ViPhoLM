# models/seneca/components.py

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .utils import len_mask, step_attention

INI = 1e-2

class MultiLayerLSTMCells(nn.Module):
    """Lấy từ rnn.py, là phiên bản unrolled của nn.LSTM."""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.cells = nn.ModuleList()
        self.cells.append(nn.LSTMCell(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.cells.append(nn.LSTMCell(hidden_size, hidden_size))
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, input_, states):
        h_prev, c_prev = states
        h_next, c_next = [], []
        for i, cell in enumerate(self.cells):
            h_i, c_i = cell(input_, (h_prev[i], c_prev[i]))
            h_next.append(h_i)
            c_next.append(c_i)
            input_ = F.dropout(h_i, p=self.dropout, training=self.training)
        return torch.stack(h_next), torch.stack(c_next)
    
    @staticmethod
    def convert(lstm_module):
        # Hàm để chuyển đổi từ một nn.LSTM tiêu chuẩn
        new_cell = MultiLayerLSTMCells(
            lstm_module.input_size, lstm_module.hidden_size,
            lstm_module.num_layers, dropout=lstm_module.dropout)
        # Sao chép trọng số
        for i, cell in enumerate(new_cell.cells):
            cell.weight_ih.data.copy_(getattr(lstm_module, f'weight_ih_l{i}'))
            cell.weight_hh.data.copy_(getattr(lstm_module, f'weight_hh_l{i}'))
            cell.bias_ih.data.copy_(getattr(lstm_module, f'bias_ih_l{i}'))
            cell.bias_hh.data.copy_(getattr(lstm_module, f'bias_hh_l{i}'))
        return new_cell
        
class ConvEncoder(nn.Module):
    """
    Encoder chung sử dụng CNN, được dùng cho cả câu và thực thể.
    """
    def __init__(self, emb_dim, n_hidden, kernel_sizes, dropout):
        super().__init__()
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, k, padding=int((k-1)/2)) for k in kernel_sizes])
        self._dropout = dropout

    def forward(self, emb_input):
        conv_in = F.dropout(emb_input.transpose(1, 2), self._dropout, training=self.training)
        # Sử dụng padding nên không cần max-over-time pooling ở đây
        outputs = [F.relu(conv(conv_in)) for conv in self._convs]
        # Concat a lo largo de la dimensión de características
        return torch.cat(outputs, dim=1).transpose(1, 2)

class LSTMPointerNet(nn.Module):
    """
    Pointer Network Decoder, logic chính từ LSTMPointerNet_entity trong extract.py.
    """
    def __init__(self, input_dim, n_hidden, n_layer, dropout, n_hop, side_dim):
        super().__init__()
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer, bidirectional=False, dropout=dropout)
        
        # Attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        
        # Entity attention parameters
        self.side_wm = nn.Parameter(torch.Tensor(side_dim, n_hidden))
        self.side_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.side_v = nn.Parameter(torch.Tensor(n_hidden))
        self._attn_ws = nn.Parameter(torch.Tensor(n_hidden, n_hidden))

        # Khởi tạo trọng số
        for p in [self._attn_wm, self._attn_wq, self.side_wm, self.side_wq, self._attn_ws]:
            init.xavier_normal_(p)
        for p in [self._attn_v, self.side_v]:
            init.uniform_(p, -INI, INI)
            
    def forward(self, sent_mem, entity_mem, ptr_in, sent_nums, entity_nums):
        """
        sent_mem: [B, max_sent, D_sent]
        entity_mem: [B, max_entity, D_entity]
        ptr_in: [B, max_decode_steps, D_sent] (teacher forcing inputs)
        """
        # LSTM Decoder
        query, _ = self._lstm(ptr_in.transpose(0, 1))
        query = query.transpose(0, 1) # [B, max_decode_steps, D_hidden]

        # Entity Attention (Glimpse operation)
        side_feat = torch.matmul(entity_mem, self.side_wm) # [B, max_entity, D_hidden]
        entity_context = self.attention(side_feat, query, self.side_v, self.side_wq, entity_nums)

        # Sentence Attention with entity context
        sent_feat = torch.matmul(sent_mem, self._attn_wm)
        score = self.attention_with_side_info(
            sent_feat, query, entity_context, self._attn_v, self._attn_wq, self._attn_ws, sent_nums
        )
        return score

    def attention(self, attention_mem, query, v, w, mem_sizes):
        # Hàm attention chung
        sum_ = attention_mem.unsqueeze(1) + torch.matmul(query, w).unsqueeze(2)
        score = torch.matmul(torch.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)).squeeze(3)
        mask = len_mask(mem_sizes, score.device).unsqueeze(1)
        norm_score = F.softmax(score.masked_fill(mask == 0, -1e18), dim=-1)
        return torch.matmul(norm_score, attention_mem)
        
    def attention_with_side_info(self, sent_mem, query, entity_context, v, w_sent, w_entity, sent_sizes):
        # Hàm attention kết hợp thông tin thực thể
        sent_part = torch.matmul(query, w_sent).unsqueeze(2)
        entity_part = torch.matmul(entity_context, w_entity).unsqueeze(2)
        sum_ = sent_mem.unsqueeze(1) + sent_part + entity_part
        score = torch.matmul(torch.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)).squeeze(3)
        mask = len_mask(sent_sizes, score.device).unsqueeze(1)
        score.masked_fill_(mask == 0, -1e18)
        return score
    
class _CopyLinear(nn.Module):
    """Lớp tính toán xác suất copy (copy gate). Lấy từ copy_summ.py."""
    def __init__(self, context_dim, state_dim, input_dim, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.Tensor(context_dim))
        self._v_s = nn.Parameter(torch.Tensor(state_dim))
        self._v_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._v_c, -INI, INI)
        init.uniform_(self._v_s, -INI, INI)
        init.uniform_(self._v_i, -INI, INI)
        if bias:
            self._b = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('_b', None)

    def forward(self, context, state, input_):
        output = (torch.matmul(context, self._v_c.unsqueeze(1))
                  + torch.matmul(state, self._v_s.unsqueeze(1))
                  + torch.matmul(input_, self._v_i.unsqueeze(1)))
        if self._b is not None:
            output = output + self._b.unsqueeze(0)
        return output

class AttentionalLSTMDecoder(nn.Module):
    """
    Lớp Decoder cơ bản với Attention. Lấy từ summ.py.
    Đây là lớp cha cho CopyLSTMDecoder.
    """
    def __init__(self, embedding, lstm_cells, attn_w, projection):
        super().__init__()
        self._embedding = embedding
        self._lstm = lstm_cells
        self._attn_w = attn_w
        self._projection = projection

    def _step(self, tok, states, attention):
        prev_states, prev_out = states
        lstm_in = torch.cat([self._embedding(tok).squeeze(1), prev_out], dim=1)
        
        new_states = self._lstm(lstm_in, prev_states)
        lstm_out = new_states[0][-1]
        
        query = torch.mm(lstm_out, self._attn_w)
        attention_context, attn_mask = attention
        context, score = step_attention(query, attention_context, attention_context, attn_mask)
        
        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))
        
        final_states = (new_states, dec_out)
        logit = torch.mm(dec_out, self._embedding.weight.t())
        
        return logit, final_states, score

    def forward(self, attention, init_states, target):
        max_len = target.size(1)
        states = init_states
        logits = []
        for i in range(max_len):
            tok = target[:, i:i+1]
            logit, states, _ = self._step(tok, states, attention)
            logits.append(logit)
        return torch.stack(logits, dim=1)

class CopyLSTMDecoder(AttentionalLSTMDecoder):
    """
    Kế thừa từ AttentionalLSTMDecoder và thêm logic copy. Lấy từ copy_summ.py.
    """
    def __init__(self, copy_linear, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._copy = copy_linear

    def _step(self, tok, states, attention):
        prev_states, prev_out = states
        lstm_in = torch.cat([self._embedding(tok).squeeze(1), prev_out], dim=1)
        
        new_states = self._lstm(lstm_in, prev_states)
        lstm_out = new_states[0][-1]
        
        query = torch.mm(lstm_out, self._attn_w)
        attention_context, attn_mask, extend_src, extend_vsize = attention
        context, score = step_attention(query, attention_context, attention_context, attn_mask)
        
        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))

        # --- Logic Copy ---
        gen_prob = self._compute_gen_prob(dec_out, extend_vsize)
        copy_prob = torch.sigmoid(self._copy(context, new_states[0][-1], lstm_in))
        
        # Kết hợp xác suất sinh và xác suất copy
        final_prob = ((1 - copy_prob) * gen_prob).scatter_add(
            dim=1,
            index=extend_src,
            src=score * copy_prob
        )
        # --- Kết thúc Logic Copy ---

        log_prob = torch.log(final_prob + 1e-8)  # Thêm epsilon để ổn định
        
        return log_prob, (new_states, dec_out), score

    def _compute_gen_prob(self, dec_out, extend_vsize, eps=1e-6):
        """Mở rộng từ điển để tính xác suất sinh từ."""
        logit = torch.mm(dec_out, self._embedding.weight.t())
        bsize, vsize = logit.size()
        if extend_vsize > vsize:
            ext_logit = torch.full((bsize, extend_vsize - vsize), eps, device=logit.device)
            gen_logit = torch.cat([logit, ext_logit], dim=1)
        else:
            gen_logit = logit
        return F.softmax(gen_logit, dim=-1)
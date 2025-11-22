""" attention functions """
from torch.nn import functional as F


def dot_attention_score(key, query):
    """[B, Tk, D], [(Bs), B, Tq, D] -> [(Bs), B, Tq, Tk]"""
    return query.matmul(key.transpose(1, 2))

def prob_normalize(score, mask):
    """ [(...), T]
    user should handle mask shape"""
    score = score.masked_fill(mask == 0, -1e18)
    norm_score = F.softmax(score, dim=-1)
    return norm_score

def attention_aggregate(value, score):
    """[B, Tv, D], [(Bs), B, Tq, Tv] -> [(Bs), B, Tq, D]"""
    output = score.matmul(value)
    return output


def step_attention(query, key, value, mem_mask=None):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    score = dot_attention_score(key, query.unsqueeze(-2))
    if mem_mask is None:
        norm_score = F.softmax(score, dim=-1)
    else:
        norm_score = prob_normalize(score, mem_mask)
    output = attention_aggregate(value, norm_score)
    return output.squeeze(-2), norm_score.squeeze(-2)

# ================= SEAL Attention (copied from models/transformer_seal/layers/seal_attention.py) =================
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEALAttention(nn.Module):
    def __init__(self, d_model, n_head, segment_size):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.segment_size = segment_size
        self.head_dim = d_model // n_head
        assert d_model % n_head == 0

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        B, L, _ = x.size()
        S = self.segment_size
        n_seg = (L + S - 1) // S  # số segment

        # 1. Chia thành các segment
        segments = x.split(S, dim=1)  # list of (batch, S, d_model)
        outputs = []

        # 2. Local attention cho từng segment
        for seg in segments:
            Q = self.w_q(seg)
            K = self.w_k(seg)
            V = self.w_v(seg)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            local_out = torch.matmul(attn_weights, V)
            outputs.append(local_out)
        local_output = torch.cat(outputs, dim=1)  # (batch, seq_len, d_model)

        # 3. Tạo segment summary (dùng token đầu tiên của mỗi segment)
        segment_summaries = torch.stack([seg[:, 0, :] for seg in segments], dim=1)  # (batch, n_seg, d_model)

        # 4. Global attention giữa các segment summary
        Qg = self.w_q(segment_summaries)
        Kg = self.w_k(segment_summaries)
        Vg = self.w_v(segment_summaries)
        attn_scores_g = torch.matmul(Qg, Kg.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights_g = F.softmax(attn_scores_g, dim=-1)
        global_out = torch.matmul(attn_weights_g, Vg)  # (batch, n_seg, d_model)

        # 5. Phát lại thông tin global cho từng token trong segment
        # (ở đây: cộng global_out vào từng token trong segment tương ứng)
        global_broadcast = []
        for i, seg in enumerate(segments):
            # global_out[:, i, :] shape: (batch, d_model)
            # expand to (batch, S, d_model)
            g = global_out[:, i, :].unsqueeze(1).expand(-1, seg.size(1), -1)
            global_broadcast.append(g)
        global_broadcast = torch.cat(global_broadcast, dim=1)

        # 6. Kết hợp local và global (cộng hoặc concat + linear)
        output = local_output + global_broadcast
        output = self.w_concat(output)
        return output
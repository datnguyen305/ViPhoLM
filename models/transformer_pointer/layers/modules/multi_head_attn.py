from torch import nn
import torch
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, head: int, d_model: int, d_kv: int):
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_q = d_kv
        self.d_kv = d_kv
        self.head = head

        self.fc_q = nn.Linear(d_model, head * d_kv)
        self.fc_k = nn.Linear(d_model, head * d_kv)
        self.fc_v = nn.Linear(d_model, head * d_kv)

    def forward(self, queries, keys, values, group_prob, attention_mask):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.head, self.d_q).permute(0, 2, 1, 3)   # (b_s, h, nq, d_q)
        k = self.fc_k(keys).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 3, 1)     # (b_s, h, nk, d_kv)
        v = self.fc_v(values).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 1, 3)   # (b_s, h, nk, d_kv)

        att = torch.matmul(q, k) / np.sqrt(self.d_kv)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            # 1. Tự động bơm thêm chiều nếu thiếu (2D, 3D -> 4D)
            while attention_mask.dim() < 4:
                attention_mask = attention_mask.unsqueeze(1)
            
            # 2. Tự động ép xẹp bớt chiều nếu bị dư (5D, 6D -> 4D)
            while attention_mask.dim() > 4:
                attention_mask = attention_mask.squeeze(1)
            
            # Lúc này mask chắc chắn là 4D: [B, 1, 1, Seq_len] hoặc [B, 1, Seq_len, Seq_len]
            att.masked_fill_(attention_mask == 0, -1e4)
        att = torch.softmax(att, dim=-1)
        att = att * group_prob
        output = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b_s, -1, self.d_model)

        return output
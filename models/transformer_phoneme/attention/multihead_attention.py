import torch
import torch.nn as nn
from models.transformer_phoneme.attention.phrasal_lexeme import Phrasal_Lexeme

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_head
        self.d_model = config.hidden_size
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_model // self.num_heads

        self.linear_q = nn.Linear(self.d_model, self.d_model)
        self.linear_k = nn.Linear(self.d_model, self.d_model)
        self.linear_v = nn.Linear(self.d_model, self.d_model)
        self.linear_out = nn.Linear(self.d_model, self.d_model)

        self.phrasal_lexeme = Phrasal_Lexeme(config)

    def forward(self, query, key, value, mask=None, causal_mask=None):
        B, S_q, _ = query.size() # Độ dài của Query
        S_k = key.size(1)       # Độ dài của Key (quan trọng!)
        S_v = value.size(1)     # Độ dài của Value

        # Linear projections
        Q = self.linear_q(query).reshape(B, S_q, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, S_q, d_k)
        K = self.linear_k(key).reshape(B, S_k, self.num_heads, self.d_k).transpose(1, 2)    # (B, num_heads, S_k, d_k)
        V = self.linear_v(value).reshape(B, S_v, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, S_v, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32, device=Q.device))
        final_mask = None
        if mask is not None and causal_mask is not None:
            # Kết hợp cả hai bằng phép toán AND (logic &)
            # mask (B, 1, 1, S_k) & casual_mask (1, 1, S_q, S_k) -> (B, 1, S_q, S_k)
            final_mask = mask & causal_mask
        elif mask is not None:
            final_mask = mask
        elif causal_mask is not None:
            final_mask = causal_mask

        if final_mask is not None:
            scores = scores.masked_fill(final_mask == 0, float('-inf'))

        # Tính xác suất cụm từ P_{i,j} (B)
        # Lưu ý: Nếu bài báo muốn mỗi head một P, bạn cần sửa Phrasal_Lexeme trả về (B, H)
        p_score = self.phrasal_lexeme(query, key) 
        
        # Chuyển đổi p_score để nhân element-wise (B, H, 1, 1) hoặc (B, 1, 1, 1)
        if p_score.dim() == 1:
            p_gate = p_score.view(B, 1, 1, 1)
        else:
            p_gate = p_score.unsqueeze(-1).unsqueeze(-1)

        attn_weights = torch.softmax(scores, dim=-1)  # (B, num_heads, S_q, S_k)

        attn_weights = attn_weights * p_gate  # (B, num_heads, S_q, S_k)

        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, S_q, S_k) * (B, num_heads, S_k, d_k) = (B, num_heads, S_q, d_k)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).reshape(B, S_q, self.d_model)  # (B, S_q, num_heads * d_k)
        output = self.linear_out(attn_output)  # (B, S_q, d_model)
        return output
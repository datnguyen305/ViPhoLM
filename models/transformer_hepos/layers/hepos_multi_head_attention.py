import torch
import torch.nn as nn
import torch.nn.functional as F

class HeposMultiHeadAttention(nn.Module):
    """
    Multi-head attention với Head-wise Positional Strides (HEPOS).
    Mỗi head chỉ attend đến các vị trí (i - h) % stride == 0.
    """
    def __init__(self, d_model, n_head, stride):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.stride = stride
        self.head_dim = d_model // n_head
        assert d_model % n_head == 0, "d_model phải chia hết cho n_head"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # q: (batch, tgt_len, d_model)
        # k, v: (batch, src_len, d_model)
        B, tgt_len, _ = q.size()
        B, src_len, _ = k.size()

        # 1. Linear projections
        Q = self.w_q(q).view(B, tgt_len, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, tgt_len, D)
        K = self.w_k(k).view(B, src_len, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, src_len, D)
        V = self.w_v(v).view(B, src_len, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, src_len, D)

        # 2. Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, tgt_len, src_len)

        # 3. HEPOS masking: mỗi head chỉ attend đến các vị trí (i - h) % stride == 0
        device = attn_scores.device
        src_indices = torch.arange(src_len, device=device)
        for h in range(self.n_head):
            # mask: (src_len,) True nếu bị mask
            mask_h = ((src_indices - h) % self.stride != 0)  # True ở vị trí không được attend
            # Kiểm tra nếu tất cả positions bị mask
            if mask_h.all():
                # Nếu tất cả bị mask, chỉ cho phép attend vào position đầu tiên
                mask_h[0] = False
            # broadcast: (B, tgt_len, src_len)
            attn_scores[:, h, :, mask_h] = float('-inf')

        # 4. Optional: apply input mask (e.g. padding)
        if mask is not None:
            # mask: (B, 1, 1, src_len) hoặc (B, 1, tgt_len, src_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, tgt_len, src_len)
        attn_output = torch.matmul(attn_weights, V)    # (B, H, tgt_len, D)

        # 5. Kết hợp các head
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, tgt_len, self.d_model)
        output = self.w_concat(attn_output)
        return output

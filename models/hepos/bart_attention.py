import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HEPOSBartAttention(nn.Module):
    """
    HEPOS Cross-Attention for BART Decoder
    Paper-faithful implementation.
    """

    def __init__(self, embed_dim, num_heads, stride, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert stride > 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.stride = stride
        self.dropout = dropout

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        hidden_states,            # decoder states (Q)
        key_value_states,         # encoder states (K,V)
        attention_mask=None,      # encoder padding mask [B,1,1,T]
        layer_head_mask=None,
        output_attentions=False,
    ):
        B, Tq, _ = hidden_states.size()
        Tk = key_value_states.size(1)
        device = hidden_states.device

        Q = self.q_proj(hidden_states)
        K = self.k_proj(key_value_states)
        V = self.v_proj(key_value_states)

        Q = Q.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        outputs = []

        for h in range(self.num_heads):
            idx = torch.arange(h, Tk, self.stride, device=device)

            Qh = Q[:, h:h+1]           # [B,1,Tq,D]
            Kh = K[:, h:h+1, idx]
            Vh = V[:, h:h+1, idx]

            scores = torch.matmul(
                Qh, Kh.transpose(-2, -1)
            ) * self.scaling

            if attention_mask is not None:
                mask_h = attention_mask[..., idx]
                scores = scores.masked_fill(mask_h == 0, -1e9)

            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)

            out = torch.matmul(attn, Vh)
            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.embed_dim)

        out = self.out_proj(out)

        return out, None

from torch import nn
from vocabs.vocab import Vocab
from ..attention.multihead_attention import MultiHeadAttention
from ..modules.ffn import FeedForwardNetwork

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        # 1. Cơ chế Attention bạn muốn sửa nằm ở đây
        self.multi_head_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        # 2. Các thành phần chuẩn của một lớp Encoder
        self.linear1 = nn.Linear(config.hidden_size, config.dim_feedforward)
        self.dropout = nn.Dropout(config.drop_prob)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.drop_prob)
        self.dropout2 = nn.Dropout(config.drop_prob)

    def forward(self, src, src_mask=None, src_causal_mask=None):
        # 1. Multi-Head Attention
        attn_input = self.norm1(src)
        attn_output = self.dropout1(self.multi_head_attn(attn_input, attn_input, attn_input, mask=src_mask, causal_mask=src_causal_mask))
        # src (B, S, hidden_size)
        output1 = src + attn_output

        # 2. Feed Forward Network
        ffn_output = self.dropout2(self.feed_forward(self.norm2(output1)))
        output = ffn_output + output1
        #  # (B, S_q, hidden-size)
        return output 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab

def to_cuda(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)

@META_ARCHITECTURE.register()
class HEPOSSummarizer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.stride_list = config.stride_list
        self.num_layers = config.num_layers
        self.vocab_size = len(vocab)
        self.padding_idx = config.padding_idx
        self.dropout = config.dropout
        self.max_tgt_len = config.max_tgt_len
        self.teacher_forcing = config.teacher_forcing

        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.padding_idx)
        self.positional_encoding = PositionalEncoding(self.d_model, dropout=self.dropout, max_len=2048)

        self.encoder_layers = nn.ModuleList([
            HEPOSEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                stride_list=self.stride_list,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])

        self.decoder_rnn = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )

        self.cross_attention = HEPOSCrossAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            stride_list=self.stride_list
        )

        self.output_layer = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, src, tgt=None):
        src_emb = self.embedding(src)
        src_emb = self.positional_encoding(src_emb)
        for layer in self.encoder_layers:
            src_emb = layer(src_emb)
        encoder_output = src_emb

        if tgt is not None and self.teacher_forcing:
            tgt_emb = self.embedding(tgt)
            tgt_emb = self.positional_encoding(tgt_emb)
            decoder_output, _ = self.decoder_rnn(tgt_emb)
            context = self.cross_attention(decoder_output, encoder_output)
            final_output = decoder_output + context
            logits = self.output_layer(final_output)
        else:
            batch_size = src.size(0)
            decoder_input = to_cuda(torch.zeros(batch_size, 1).long())
            hidden = None
            logits = []
            for _ in range(self.max_tgt_len):
                emb = self.embedding(decoder_input)
                emb = self.positional_encoding(emb)
                decoder_output, hidden = self.decoder_rnn(emb, hidden)
                context = self.cross_attention(decoder_output, encoder_output)
                final_output = decoder_output + context
                logit = self.output_layer(final_output.squeeze(1))
                logits.append(logit)
                decoder_input = logit.argmax(-1).unsqueeze(1)
            logits = torch.stack(logits, dim=1)

        return logits

class HEPOSEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, stride_list, dropout=0.1):
        super().__init__()
        self.self_attn = HEPOSSelfAttention(d_model, num_heads, stride_list)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.norm(x + self.dropout(ffn_output))

class HEPOSSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, stride_list):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.stride_list = stride_list

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        outputs = []
        for h in range(self.num_heads):
            stride = self.stride_list[h]
            k_strided = k[:, h, ::stride, :]
            v_strided = v[:, h, ::stride, :]
            scores = torch.matmul(q[:, h], k_strided.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_strided)
            outputs.append(out)

        output = torch.stack(outputs, dim=1).transpose(1, 2).reshape(B, L, self.d_model)
        return self.out_proj(output)

class HEPOSCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, stride_list):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.stride_list = stride_list
        self.head_dim = d_model // num_heads

        assert len(stride_list) == num_heads, "Cần stride cho mỗi head"

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, decoder_input, encoder_output):
        B, T, _ = decoder_input.shape
        _, S, _ = encoder_output.shape

        Q = self.query_proj(decoder_input)
        K = self.key_proj(encoder_output)
        V = self.value_proj(encoder_output)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        outputs = []
        for h in range(self.num_heads):
            stride = self.stride_list[h]
            K_h = K[:, h, ::stride, :]
            V_h = V[:, h, ::stride, :]
            scores = torch.matmul(Q[:, h], K_h.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, V_h)
            outputs.append(out)

        output = torch.stack(outputs, dim=1).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(output)

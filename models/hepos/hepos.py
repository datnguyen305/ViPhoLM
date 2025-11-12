import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab


class PositionalEncoding(nn.Module):
    """Lớp Positional Encoding sin/cos chuẩn."""
    def __init__(self, d_model, dropout=0.1, max_len=10240):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Các hàm tiện ích cho Sinkhorn Attention
def gumbel_sinkhorn(log_alpha, temp=0.7, n_iters=7):
    """Thuật toán Gumbel-Sinkhorn để tạo ma trận hoán vị."""
    gumbel = -torch.log(-torch.log(torch.rand_like(log_alpha) + 1e-20) + 1e-20)
    log_alpha = (log_alpha + gumbel) / temp
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)

def bucket(t, bucket_size):
    # t: (B, H, L, D)
    B, H, L, D = t.shape
    pad_len = (bucket_size - (L % bucket_size)) % bucket_size
    if pad_len > 0:
        pad = torch.zeros(B, H, pad_len, D, device=t.device, dtype=t.dtype)
        t = torch.cat([t, pad], dim=2)
    L = t.size(2)
    num_buckets = L // bucket_size
    # return (B, H, N, S, D)
    return t.view(B, H, num_buckets, bucket_size, D)


def unbucket(t):
    # t: (B, H, N, S, D) -> (B, H, N*S, D)
    B, H, N, S, D = t.shape
    return t.reshape(B, H, N * S, D)


class SinkhornSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.block_size = block_size  # b_s trong paper
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Sorting network parameters
        self.sorting_net = nn.Linear(self.head_dim, self.head_dim)

    def forward(self, x, padding_mask=None):
        B, L, _ = x.shape
        
        # Project Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Chia thành blocks
        num_blocks = (L + self.block_size - 1) // self.block_size
        pad_len = num_blocks * self.block_size - L
        
        if pad_len > 0:
            # Pad để chia đều
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
        
        # Reshape thành blocks: (B, H, num_blocks, block_size, head_dim)
        q_blocks = q.view(B, self.num_heads, num_blocks, self.block_size, self.head_dim)
        k_blocks = k.view(B, self.num_heads, num_blocks, self.block_size, self.head_dim)
        v_blocks = v.view(B, self.num_heads, num_blocks, self.block_size, self.head_dim)
        
        # Tính block representatives
        q_repr = q_blocks.mean(dim=3)  # (B, H, num_blocks, head_dim)
        k_repr = k_blocks.mean(dim=3)
        
        # Sinkhorn sorting (simplified)
        similarity = torch.matmul(q_repr, k_repr.transpose(-2, -1)) * self.scale
        sort_matrix = gumbel_sinkhorn(similarity)  # (B, H, num_blocks, num_blocks)
        
        # Reorder blocks
        k_blocks_flat = k_blocks.flatten(3)  # (B, H, num_blocks, block_size*head_dim)
        v_blocks_flat = v_blocks.flatten(3)
        
        k_sorted = torch.matmul(sort_matrix, k_blocks_flat)
        v_sorted = torch.matmul(sort_matrix, v_blocks_flat)
        
        k_sorted = k_sorted.view(B, self.num_heads, num_blocks, self.block_size, self.head_dim)
        v_sorted = v_sorted.view(B, self.num_heads, num_blocks, self.block_size, self.head_dim)
        
        # Attend trong block và block kế
        outputs = []
        for i in range(num_blocks):
            q_block = q_blocks[:, :, i]  # (B, H, block_size, head_dim)
            
            # Keys: current block + sorted neighbor
            if i < num_blocks - 1:
                k_local = torch.cat([
                    k_blocks[:, :, i],      # same block
                    k_sorted[:, :, i+1]     # sorted next block
                ], dim=2)  # (B, H, 2*block_size, head_dim)
                
                v_local = torch.cat([
                    v_blocks[:, :, i],
                    v_sorted[:, :, i+1]
                ], dim=2)
            else:
                k_local = k_blocks[:, :, i]
                v_local = v_blocks[:, :, i]
            
            # Local attention: O(block_size * 2*block_size) = O(2*b_s²)
            scores = torch.matmul(q_block, k_local.transpose(-2, -1)) * self.scale
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            
            out = torch.matmul(attn, v_local)  # (B, H, block_size, head_dim)
            outputs.append(out)
        
        # Concatenate blocks
        output = torch.cat(outputs, dim=2)  # (B, H, num_blocks*block_size, head_dim)
        
        # Remove padding
        if pad_len > 0:
            output = output[:, :, :L]
        
        # Reshape và project
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(output)


class HEPOSCrossAttention(nn.Module):
    """
    Triển khai logic HEPOS cho Cross-Attention, không phụ thuộc Fairseq.
    Logic: Mỗi head chỉ attend đến một tập con các token của encoder theo bước nhảy (stride).
    """
    def __init__(self, d_model, num_heads, stride_size=4):
        super().__init__()
        self.num_heads = num_heads
        self.stride_size = stride_size
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, key_padding_mask=None):
        B, T, _ = query.shape # (Batch, Target_Len, Dim)
        _, S, _ = key.shape   # (Batch, Source_Len, Dim)

        q = self.query_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (B, H, T, S)

        # --- Logic cốt lõi của HEPOS ---
        indices = torch.arange(S, device=scores.device).view(1, 1, 1, S)
        head_indices = torch.arange(self.num_heads, device=scores.device).view(1, self.num_heads, 1, 1)
        
        # Mask những vị trí không thỏa mãn điều kiện (i - h) % stride == 0
        hepos_mask = (indices - head_indices) % self.stride_size != 0
        scores = scores.masked_fill(hepos_mask, float('-inf'))
        # -----------------------------

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.out_proj(context)

# Các lớp thành phần cho Encoder và Decoder

class SinkhornEncoderLayer(nn.Module):
    """Một lớp Encoder hoàn chỉnh sử dụng Sinkhorn Self-Attention."""
    def __init__(self, d_model, num_heads, bucket_size, dropout=0.1):
        super().__init__()
        self.self_attn = SinkhornSelfAttention(d_model, num_heads, bucket_size, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Sub-layer 1: Self-Attention
        attn_output = self.self_attn(src, padding_mask=src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        # Sub-layer 2: Feed Forward
        ffn_output = self.ffn(src)
        src = self.norm2(src + self.dropout(ffn_output))
        return src

class HeposDecoderLayer(nn.Module):
    """Một lớp Decoder hoàn chỉnh sử dụng HEPOS Cross-Attention."""
    def __init__(self, d_model, num_heads, hepos_stride_size, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = HEPOSCrossAttention(d_model, num_heads, stride_size=hepos_stride_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        # Sub-layer 1: Masked Self-Attention
        attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_output))
        # Sub-layer 2: HEPOS Cross-Attention
        cross_attn_output = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn_output))
        # Sub-layer 3: Feed Forward
        ffn_output = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_output))
        return tgt


@META_ARCHITECTURE.register()
class HEPOSBaselineSummarizer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        # Lấy các tham số từ file config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_encoder_layers = config.num_encoder_layers
        self.num_decoder_layers = config.num_decoder_layers
        self.vocab_size = len(vocab)
        self.padding_idx = vocab.pad_idx
        self.dropout = config.dropout
        
        # Tham số riêng cho baseline
        self.sinkhorn_bucket_size = config.sinkhorn_bucket_size
        self.hepos_stride_size = config.hepos_stride_size
        
        # Các module chung
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.padding_idx)
        self.positional_encoding = PositionalEncoding(self.d_model, dropout=self.dropout)
        
        # Encoder: Một chuỗi các lớp SinkhornEncoderLayer
        self.encoder_layers = nn.ModuleList([
            SinkhornEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                bucket_size=self.sinkhorn_bucket_size,
                dropout=self.dropout
            ) for _ in range(self.num_encoder_layers)
        ])

        # Decoder: Một chuỗi các lớp HeposDecoderLayer
        self.decoder_layers = nn.ModuleList([
            HeposDecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                hepos_stride_size=self.hepos_stride_size,
                dropout=self.dropout
            ) for _ in range(self.num_decoder_layers)
        ])
        
        self.output_layer = nn.Linear(self.d_model, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, tgt_mask=None):
        """
        Args:
            src: (B, S_len) - input sequence
            tgt: (B, T_len) - target sequence (with <bos>)
        
        Returns:
            logits: (B, T_len, vocab_size)
            loss: scalar tensor (nếu training)
        """
        # --- ENCODER ---
        src_emb = self.positional_encoding(self.embedding(src) * math.sqrt(self.d_model))
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask=src_padding_mask)
            
        # --- DECODER ---
        tgt_emb = self.positional_encoding(self.embedding(tgt) * math.sqrt(self.d_model))
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(
                decoder_output, encoder_output, 
                tgt_mask=tgt_mask, 
                memory_key_padding_mask=src_padding_mask
            )

        logits = self.output_layer(decoder_output)  # (B, T_len, vocab_size)
        
        # Tính loss nếu đang training
        if self.training:
            # Shift targets: dự đoán token tiếp theo
            # logits: (B, T_len, V) -> (B, T_len-1, V)
            # tgt: (B, T_len) -> (B, T_len-1)
            logits_for_loss = logits[:, :-1, :].contiguous()  # Bỏ token cuối
            targets = tgt[:, 1:].contiguous()  # Bỏ <bos>, shift left
            
            # Reshape để tính loss
            loss = self.criterion(
                logits_for_loss.view(-1, self.vocab_size),  # (B*(T-1), V)
                targets.view(-1)  # (B*(T-1),)
            )
            return logits, loss
        else:
            return logits, None
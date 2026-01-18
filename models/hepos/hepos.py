import math
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
from typing import Dict
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

# sinkhorn utilities
def sinkhorn(log_alpha, n_iters=5):
    """
    log_alpha: (B, N, N)
    return: doubly-stochastic matrix (NO grad)
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)

# sinkhorn self attention 
class SinkhornSelfAttention(nn.Module):
    """
    Efficient Sinkhorn Self-Attention (Encoder)
    - Block-based
    - Hard permutation (argmax)
    - No gradient through permutation
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int = 256,
        sinkhorn_iters: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.sinkhorn_iters = sinkhorn_iters
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (src_len, B, D)
        """
        src_len, B, D = x.shape
        assert src_len % self.block_size == 0, \
            f"src_len ({src_len}) must be divisible by block_size ({self.block_size})"

        num_blocks = src_len // self.block_size

        # ============================================================
        # 1️⃣ BLOCK REPRESENTATION (mean pooling)
        # ============================================================
        x_blocks = (
            x.reshape(num_blocks, self.block_size, B, D)
             .mean(dim=1)                      # (num_blocks, B, D)
             .transpose(0, 1)                  # (B, num_blocks, D)
        )

        # ============================================================
        # 2️⃣ SINKHORN PERMUTATION (NO GRAD)
        # ============================================================
        with torch.no_grad():
            q_blk = self.q_proj(x_blocks)
            k_blk = self.k_proj(x_blocks)

            logits = torch.matmul(
                q_blk, k_blk.transpose(-2, -1)
            ) / math.sqrt(D)                   # (B, num_blocks, num_blocks)

            P = sinkhorn(logits, self.sinkhorn_iters)
            perm = P.argmax(dim=-1)             # (B, num_blocks)

        # ============================================================
        # 3️⃣ HARD BLOCK REORDER (FAST INDEXING)
        # ============================================================
        x_blocks_full = (
            x.reshape(num_blocks, self.block_size, B, D)
             .permute(2, 0, 1, 3)               # (B, num_blocks, block, D)
        )

        x_perm_blocks = torch.stack(
            [x_blocks_full[b, perm[b]] for b in range(B)],
            dim=0
        )                                       # (B, num_blocks, block, D)

        x_perm = (
            x_perm_blocks
            .permute(1, 2, 0, 3)
            .reshape(src_len, B, D)
        )

        # ============================================================
        # 4️⃣ LOCAL + NEIGHBOR SELF-ATTENTION
        # ============================================================
        q = self.q_proj(x_perm)
        k = self.k_proj(x_perm)
        v = self.v_proj(x_perm)

        q = q.transpose(0, 1).reshape(B, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.transpose(0, 1).reshape(B, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.transpose(0, 1).reshape(B, src_len, self.num_heads, self.head_dim).transpose(1, 2)


        out = torch.zeros_like(q)   # (B, H, src_len, head_dim)

        for b in range(num_blocks):
            s = b * self.block_size
            e = s + self.block_size

            q_blk = q[:, :, s:e]          # (B, H, block, D)
            k_blk = k[:, :, s:e]
            v_blk = v[:, :, s:e]

            scores = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * self.scale
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out_blk = torch.matmul(attn, v_blk)  # (B, H, block, D)

            out[:, :, s:e] = out_blk

        out = (
            out.transpose(1, 2)                 # (B, src_len, H, D)
            .reshape(B, src_len, self.embed_dim)
            .transpose(0, 1)                 # (src_len, B, D)
        )
        return self.out_proj(out)


class SinkhornEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.attn = SinkhornSelfAttention(embed_dim, num_heads, block_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class SinkhornEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, block_size, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.layers = nn.ModuleList([
            SinkhornEncoderLayer(embed_dim, num_heads, block_size)
            for _ in range(num_layers)
        ])
        self.block_size = block_size
        self.pad_id = pad_id

    def forward(self, src_tokens):
        """
        src_tokens: (B, src_len)
        """
        B, src_len = src_tokens.shape

        # 🔥 PAD TO MULTIPLE OF block_size
        pad_len = (self.block_size - src_len % self.block_size) % self.block_size
        if pad_len > 0:
            pad = torch.full(
                (B, pad_len),
                self.pad_id,
                dtype=src_tokens.dtype,
                device=src_tokens.device
            )
            src_tokens = torch.cat([src_tokens, pad], dim=1)

        x = self.embed(src_tokens).transpose(0, 1)

        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)


        return x


class HeposCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, stride, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.stride = stride
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def build_mask(self, tgt_len, src_len, device):
        mask = torch.zeros(self.num_heads, tgt_len, src_len, device=device, dtype=torch.bool)
        pos = torch.arange(src_len, device=device)
        for h in range(self.num_heads):
            mask[h, :, ((pos - h) % self.stride) == 0] = True
        return mask

    def forward(self, query, key, value):
        """
        query: (tgt_len, B, D)
        key/value: (src_len, B, D)
        """
        tgt_len, B, _ = query.shape
        src_len = key.size(0)
        H = self.num_heads
        D = self.head_dim
        device = query.device

        # ---- project ----
        q = self.q_proj(query).transpose(0, 1)      # (B, tgt_len, D)
        k = self.k_proj(key).transpose(0, 1)        # (B, src_len, D)
        v = self.v_proj(value).transpose(0, 1)      # (B, src_len, D)

        q = q.reshape(B, tgt_len, H, D).transpose(1, 2)  # (B, H, tgt_len, D)
        k = k.reshape(B, src_len, H, D).transpose(1, 2)  # (B, H, src_len, D)
        v = v.reshape(B, src_len, H, D).transpose(1, 2)

        out = torch.zeros(B, H, tgt_len, D, device=device)

        # ---- HEPOS STRIDED ATTENTION (NO FULL QKᵀ) ----
        for h in range(H):
            idx = torch.arange(h, src_len, self.stride, device=device)
            k_h = k[:, h, idx]          # (B, src_len/stride, D)
            v_h = v[:, h, idx]

            q_h = q[:, h]               # (B, tgt_len, D)

            scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out[:, h] = torch.matmul(attn, v_h)

        # ---- merge heads ----
        out = (
            out.transpose(1, 2)                 # (B, tgt_len, H, D)
              .reshape(B, tgt_len, self.embed_dim)
              .transpose(0, 1)                 # (tgt_len, B, D)
        )

        return self.out_proj(out)



class HeposDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, stride):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attn = HeposCrossAttention(embed_dim, num_heads, stride)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_out):
        tgt_len = x.size(0)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=x.device),
            diagonal=1
        ).bool()

        x = self.norm1(x + self.self_attn(x, x, x, attn_mask=causal_mask)[0])
        x = self.norm2(x + self.cross_attn(x, encoder_out, encoder_out))
        x = self.norm3(x + self.ffn(x))
        return x


class HeposDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, stride):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            HeposDecoderLayer(embed_dim, num_heads, stride)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt_tokens, encoder_out):
        x = self.embed(tgt_tokens).transpose(0, 1)
        for layer in self.layers:
            x = layer(x, encoder_out)
        return self.lm_head(x.transpose(0, 1))

@META_ARCHITECTURE.register()
class HeposModel(nn.Module):
    """
    HEPOS baseline for long document summarization
    """

    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.d_model = config.d_model
        vocab_size = len(vocab)

        self.encoder = SinkhornEncoder(
            vocab_size=vocab_size,
            embed_dim=config.d_model,
            num_layers=config.encoder.num_layers,
            num_heads=config.encoder.num_heads,
            block_size=config.encoder.block_size,
        )

        self.decoder = HeposDecoder(
            vocab_size=vocab_size,
            embed_dim=config.d_model,
            num_layers=config.decoder.num_layers,
            num_heads=config.decoder.num_heads,
            stride=config.decoder.hepos_stride,
        )

    def forward(self, input_ids, labels=None):
        """
        input_ids: Tensor[B, src_len]
        labels: Tensor[B, tgt_len] or None
        """

        # ---- Encoder ----
        enc_out = self.encoder(input_ids)

        # ---- Decoder ----
        if labels is not None:
            # teacher forcing
            logits = self.decoder(labels[:, :-1], enc_out)
        else:
            logits = self.decoder(input_ids, enc_out)

        # ---- Loss ----
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1)
            )

        return logits, loss

    def predict(
        self,
        input_ids: torch.Tensor,
        max_len: int = 512,
        bos_id: int = 1,
        eos_id: int = 2
    ):
        """
        input_ids: Tensor[B, src_len]
        return: Tensor[B, generated_len]
        """
        self.eval()

        B = input_ids.size(0)
        device = input_ids.device

        # ---- Encode once ----
        enc_out = self.encoder(input_ids)

        # ---- Init decoder input ----
        ys = torch.full(
            (B, 1),
            bos_id,
            dtype=torch.long,
            device=device
        )

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self.decoder(ys, enc_out)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

            ys = torch.cat([ys, next_token], dim=1)

            finished |= (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        return ys
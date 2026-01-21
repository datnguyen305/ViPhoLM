import math
import torch
import torch.nn as nn
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab


# ============================================================
# LayerNorm
# ============================================================
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


# ============================================================
# Feed Forward
# ============================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Longformer-style Local Self Attention (Encoder)
# ============================================================
class LongformerSelfAttention(nn.Module):
    """
    Sliding-window self-attention (SAFE, no full O(L^2))
    """
    def __init__(self, d_model, n_head, window_size, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.window = window_size

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.size()
        outputs = []

        for i in range(0, L, self.window):
            s = max(0, i - self.window)
            e = min(L, i + self.window)

            q = x[:, i:i + self.window]
            k = x[:, s:e]
            v = x[:, s:e]

            out, _ = self.attn(q, k, v)
            outputs.append(out)

        return torch.cat(outputs, dim=1)


# ============================================================
# HEPOS Cross Attention (Encoder → Decoder)
# ============================================================
class HeposMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, stride, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        assert stride > 0

        self.n_head = n_head
        self.d_head = d_model // n_head
        self.stride = stride

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, src_mask=None):
        # q: (B, Tq, D), k/v: (B, Tk, D)
        B, Tq, _ = q.size()
        Tk = k.size(1)
        device = q.device

        Q = self.w_q(q).view(B, Tq, self.n_head, self.d_head).transpose(1, 2)
        K = self.w_k(k).view(B, Tk, self.n_head, self.d_head).transpose(1, 2)
        V = self.w_v(v).view(B, Tk, self.n_head, self.d_head).transpose(1, 2)

        outputs = []

        for h in range(self.n_head):
            idx = torch.arange(h, Tk, self.stride, device=device)

            Qh = Q[:, h:h+1]
            Kh = K[:, h:h+1, idx]
            Vh = V[:, h:h+1, idx]

            scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.d_head)

            if src_mask is not None:
                mask_h = src_mask[..., idx]
                scores = scores.masked_fill(
                    mask_h == 0,
                    torch.finfo(scores.dtype).min
                )

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, Vh)
            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.w_o(out)


# ============================================================
# Encoder / Decoder Layers
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout, window):
        super().__init__()
        self.attn = LongformerSelfAttention(d_model, n_head, window, dropout)
        self.norm1 = LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout, stride):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.norm1 = LayerNorm(d_model)

        self.cross_attn = HeposMultiHeadAttention(d_model, n_head, stride, dropout)
        self.norm2 = LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm3 = LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc, src_mask):
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device),
            diagonal=1
        ).bool()

        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc, enc, src_mask)))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


# ============================================================
# MAIN MODEL
# ============================================================
@META_ARCHITECTURE.register()
class HeposLongformerSummarizer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

        self.d_model = config.d_model
        self.pad = vocab.pad_idx

        self.tok_emb = nn.Embedding(vocab.vocab_size, self.d_model, padding_idx=self.pad)
        self.pos_emb = nn.Embedding(config.max_len, self.d_model)
        self.scale = math.sqrt(self.d_model)

        self.encoder = nn.ModuleList([
            EncoderLayer(
                self.d_model,
                config.ffn_hidden,
                config.n_head,
                config.drop_prob,
                config.encoder.window_size,
            )
            for _ in range(config.encoder.n_layers)
        ])

        self.decoder = nn.ModuleList([
            DecoderLayer(
                self.d_model,
                config.ffn_hidden,
                config.n_head,
                config.drop_prob,
                config.decoder.hepos_stride,
            )
            for _ in range(config.decoder.n_layers)
        ])

        self.fc = nn.Linear(self.d_model, vocab.vocab_size)

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.pad,
            label_smoothing=0.05,
        )

    # --------------------------------------------------------
    def make_src_mask(self, src):
        return (src != self.pad).unsqueeze(1).unsqueeze(2)

    # --------------------------------------------------------
    def forward(self, src, trg):
        """
        TRAINING (teacher forcing with SHIFT-RIGHT)
        """
        B, S = src.size()
        T = trg.size(1)

        src = src[:, :self.config.max_len]
        trg = trg[:, :self.config.max_len]

        # ---- Encoder embedding ----
        pos_s = torch.arange(src.size(1), device=src.device)
        enc = self.tok_emb(src) * self.scale + self.pos_emb(pos_s)

        src_mask = self.make_src_mask(src)
        for layer in self.encoder:
            enc = layer(enc)

        # ---- SHIFT RIGHT ----
        bos = torch.full(
            (B, 1),
            self.vocab.bos_idx,
            device=trg.device,
            dtype=torch.long
        )
        dec_in = torch.cat([bos, trg[:, :-1]], dim=1)

        pos_t = torch.arange(dec_in.size(1), device=trg.device)
        dec = self.tok_emb(dec_in) * self.scale + self.pos_emb(pos_t)

        for layer in self.decoder:
            dec = layer(dec, enc, src_mask)

        logits = self.fc(dec)
        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            trg.reshape(-1)
        )

        return logits, loss

    # --------------------------------------------------------
    @torch.no_grad()
    def predict(self, src, max_len=256):
        """
        GREEDY DECODING
        """
        self.eval()
        device = src.device

        src = src[:, :self.config.max_len]
        B, S = src.size()

        pos_s = torch.arange(S, device=device)
        enc = self.tok_emb(src) * self.scale + self.pos_emb(pos_s)

        src_mask = self.make_src_mask(src)
        for layer in self.encoder:
            enc = layer(enc)

        ys = torch.full(
            (B, 1),
            self.vocab.bos_idx,
            device=device,
            dtype=torch.long
        )

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            pos_t = torch.arange(ys.size(1), device=device)
            dec = self.tok_emb(ys) * self.scale + self.pos_emb(pos_t)

            for layer in self.decoder:
                dec = layer(dec, enc, src_mask)

            logits = self.fc(dec)

            # tránh EOS quá sớm
            if ys.size(1) == 1:
                logits[:, -1, self.vocab.eos_idx] = -1e9

            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

            finished |= (next_token.squeeze(1) == self.vocab.eos_idx)
            if finished.all():
                break

        return ys[:, 1:]

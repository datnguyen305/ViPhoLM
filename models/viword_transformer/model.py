import math
import torch
import torch.nn as nn
from vocabs.hierarchy_vocab import Hierarchy_Vocab
from builders.model_builder import META_ARCHITECTURE
from .attention import PhrasalLexemeAttention


class LocalAttention(nn.Module):
    """
    Input:
        x    : (B*N, S, D)
        mask : (B*N, S) bool
    Output:
        sent : (B*N, D)
    """
    def __init__(self, n_head, hidden_size, dropout=0.1):
        super().__init__()
        assert hidden_size % n_head == 0

        self.n_head = n_head
        self.d_head = hidden_size // n_head

        self.phrasal_attn = PhrasalLexemeAttention(
            head=n_head,
            d_model=hidden_size,
            d_q=self.d_head,
            d_kv=self.d_head,
        )

        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        BN, S, D = x.size()
        mask = mask.bool()

        has_token = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[~has_token, 0] = True

        safe_mask = safe_mask.cuda()

        attn, _ = self.phrasal_attn(
            x,
            attention_mask=safe_mask
        ) # (BN, H, S, S)

        v = self.v_proj(x)
        v = v.view(BN, S, self.n_head, self.d_head).transpose(1, 2)

        ctx = torch.matmul(attn, v)  # (BN, H, S, d_head)
        ctx = ctx.transpose(1, 2).contiguous().view(BN, S, D)

        x = self.norm(x + self.dropout(self.o_proj(ctx)))

        mask_f = mask.unsqueeze(-1).float()
        sent = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        sent[~has_token] = 0.0

        return sent


class FeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class HierarchicalEncoderLayer(nn.Module):
    def __init__(self, hidden_size, n_head, ffn_hidden, dropout):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            hidden_size, n_head,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = FeedForward(hidden_size, ffn_hidden, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (B, N, D)
        # mask: (B, N) bool
        key_padding_mask = ~mask

        out, _ = self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask
        )

        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class HierarchicalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList([
            HierarchicalEncoderLayer(
                config.hidden_size,
                config.n_head,
                config.ffn_hidden,
                config.drop_prob,
            )
            for _ in range(config.n_layers)
        ])

    def forward(self, sent, sent_mask):
        # sent: (B, N, D)
        # sent_mask: (B, N)
        for layer in self.layers:
            sent = layer(sent, sent_mask)
        return sent, sent_mask


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_head,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, memory_mask):
        key_padding_mask = ~memory_mask
        out, _ = self.attn(
            x, memory, memory,
            key_padding_mask=key_padding_mask
        )
        return self.norm(x + self.dropout(out))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, dropout):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_head,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attn = CrossAttention(d_model, n_head, dropout)
        self.ffn = FeedForward(d_model, ffn_hidden, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, memory_mask, tgt_key_padding_mask=None):
        out, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = self.norm1(x + self.dropout(out))

        x = self.cross_attn(x, memory, memory_mask)

        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

@META_ARCHITECTURE.register()
class ViWordTransformerModel(nn.Module):
    def __init__(self, config, vocab: Hierarchy_Vocab):
        super().__init__()
        self.vocab = vocab
        self.d_model = config.d_model
        self.MAX_LEN = vocab.max_target_length + 2  # + BOS, EOS

        self.embed = nn.Embedding(
            vocab.vocab_size, self.d_model, padding_idx=vocab.pad_idx
        )

        # token-level PE (S)
        self.pe_src = PositionalEncoding(self.d_model, vocab.max_sentence_length)
        self.pe_tgt = PositionalEncoding(self.d_model, self.MAX_LEN)

        self.local_attn = LocalAttention(
            n_head=config.encoder.n_head,
            hidden_size=config.d_model,
            dropout=config.encoder.drop_prob,
        )

        # sentence-level PE (N)
        self.pe_sentence = PositionalEncoding(self.d_model, vocab.max_sentences)

        self.encoder = HierarchicalEncoder(config.encoder)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                self.d_model,
                config.decoder.n_head,
                config.decoder.ffn_hidden,
                config.decoder.drop_prob,
            )
            for _ in range(config.decoder.n_layers)
        ])

        self.fc_out = nn.Linear(self.d_model, vocab.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def _encode(self, src):
        """
        src: (B, N, S)
        """
        B, N, S = src.size()
        src_mask = (src != self.vocab.pad_idx)   # (B, N, S)

        # ---- Token embedding ----
        x = self.embed(src.view(B * N, S))       # (B*N, S, D)
        x = self.pe_src(x)

        # ---- Local attention: token → sentence ----
        sent = self.local_attn(
            x,
            src_mask.view(B * N, S)
        ).view(B, N, -1)                         # (B, N, D)

        sent_mask = src_mask.any(dim=-1)         # (B, N)

        # ---- Sentence positional encoding ----
        sent = self.pe_sentence(sent)

        # ---- Sentence-level encoder ----
        memory, mem_mask = self.encoder(sent, sent_mask)

        return memory, mem_mask

    def forward(self, src, trg):
        trg_in  = trg[:, :-1]
        trg_out = trg[:, 1:]

        memory, mem_mask = self._encode(src)

        x = self.pe_tgt(self.embed(trg_in))

        T = trg_in.size(1)
        causal = torch.triu(
            torch.ones(T, T, device=trg.device), 1
        ).bool()

        tgt_key_padding_mask = (trg_in == self.vocab.pad_idx)  # (B, T-1)  True = ignore

        for layer in self.decoder_layers:
            x = layer(x, memory, causal, mem_mask, tgt_key_padding_mask)

        logits = self.fc_out(x)
        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            trg_out.reshape(-1),
        )

        return logits, loss

    @torch.no_grad()
    def predict(self, src):
        memory, mem_mask = self._encode(src)
        B = src.size(0)
        device = src.device

        ys = torch.full(
            (B, 1),
            self.vocab.bos_idx,
            device=device,
            dtype=torch.long
        )

        for _ in range(self.MAX_LEN):
            x = self.pe_tgt(self.embed(ys))

            T = ys.size(1)
            causal = torch.triu(
                torch.ones(T, T, device=device), 1
            ).bool()

            # No tgt_key_padding_mask needed during inference —
            # all tokens in ys are real (generated), never padding
            for layer in self.decoder_layers:
                x = layer(x, memory, causal, mem_mask)

            next_tok = self.fc_out(x[:, -1]).argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)

            if (next_tok == self.vocab.eos_idx).all():
                break

        return ys
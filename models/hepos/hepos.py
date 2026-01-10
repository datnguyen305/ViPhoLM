import torch
import math
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

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
    def __init__(self, d_model, hidden, drop_prob):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


# ============================================================
# FULL Multi-Head Attention (SAFE)
# ============================================================

class FullMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Tq, _ = q.size()
        Tk = k.size(1)
        D = self.head_dim

        Q = self.w_q(q).view(B, Tq, self.n_head, D).transpose(1, 2)
        K = self.w_k(k).view(B, Tk, self.n_head, D).transpose(1, 2)
        V = self.w_v(v).view(B, Tk, self.n_head, D).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.w_o(out)


# ============================================================
# HEPOS Cross-Attention (ONLY encoder-decoder)
# ============================================================

class HeposMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, stride, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        assert stride > 0

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.stride = stride

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, src_mask=None):
        B, Tq, _ = q.size()
        Tk = k.size(1)
        D = self.head_dim
        device = q.device

        Q = self.w_q(q).view(B, Tq, self.n_head, D).transpose(1, 2)
        K = self.w_k(k).view(B, Tk, self.n_head, D).transpose(1, 2)
        V = self.w_v(v).view(B, Tk, self.n_head, D).transpose(1, 2)

        outputs = []

        for h in range(self.n_head):
            idx = torch.arange(h, Tk, self.stride, device=device)

            Qh = Q[:, h:h+1]
            Kh = K[:, h:h+1, idx]
            Vh = V[:, h:h+1, idx]

            scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(D)

            if src_mask is not None:
                mask_h = src_mask[..., idx]
                scores = scores.masked_fill(mask_h == 0, -1e9)

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
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.attn = FullMultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout1(self.attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, stride):
        super().__init__()
        self.self_attn = FullMultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attn = HeposMultiHeadAttention(d_model, n_head, stride)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, x, enc, trg_mask, src_mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, trg_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, enc, enc, src_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


# ============================================================
# Embedding
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        pe = torch.zeros(max_len, d_model, device=device)
        pos = torch.arange(0, max_len, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos = PositionalEncoding(d_model, max_len, device)
        self.scale = math.sqrt(d_model)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        return self.drop(self.tok(x) * self.scale + self.pos(x))


# ============================================================
# Encoder / Decoder
# ============================================================

class Encoder(nn.Module):
    def __init__(self, config, vocab, device):
        super().__init__()
        self.emb = TransformerEmbedding(vocab.vocab_size, config.d_model, config.max_len, config.drop_prob, device)
        self.layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.ffn_hidden, config.n_head, config.drop_prob)
            for _ in range(config.n_layers)
        ])

    def forward(self, x, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, config, vocab, device):
        super().__init__()
        self.emb = TransformerEmbedding(vocab.vocab_size, config.d_model, config.max_len, config.drop_prob, device)
        self.layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.ffn_hidden, config.n_head, config.drop_prob, config.stride)
            for _ in range(config.n_layers)
        ])
        self.fc = nn.Linear(config.d_model, vocab.vocab_size)

    def forward(self, trg, enc, trg_mask, src_mask):
        x = self.emb(trg)
        for layer in self.layers:
            x = layer(x, enc, trg_mask, src_mask)
        return self.fc(x)


# ============================================================
# MAIN MODEL
# ============================================================

@META_ARCHITECTURE.register()
class HEPOSBaselineSummarizer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.d_model = config.encoder.d_model

        self.encoder = Encoder(config.encoder, vocab, config.device)
        self.decoder = Decoder(config.decoder, vocab, config.device)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def make_src_mask(self, src):
        return (src != self.vocab.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        T = trg.size(1)
        pad = (trg != self.vocab.pad_idx).unsqueeze(1).unsqueeze(2)
        causal = torch.tril(torch.ones(T, T, device=trg.device)).bool().unsqueeze(0).unsqueeze(1)
        return pad & causal

    def forward(self, src, trg):
        src = src[:, :self.config.encoder.max_len]
        trg = trg[:, :self.config.decoder.max_len]

        B = trg.size(0)
        bos = torch.full((B, 1), self.vocab.bos_idx, device=trg.device)
        dec_in = torch.cat([bos, trg[:, :-1]], dim=1)

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(dec_in)

        enc = self.encoder(src, src_mask)
        out = self.decoder(dec_in, enc, trg_mask, src_mask)

        if torch.isnan(out).any():
            raise RuntimeError("NaN in decoder output")

        loss = self.loss_fn(out.reshape(-1, out.size(-1)), trg.reshape(-1))
        if torch.isnan(loss):
            raise RuntimeError("NaN loss")

        return out, loss
    @torch.no_grad()
    def predict(self, src, max_len=None, beam_size=1):
        """
        Generate summary (greedy or beam search)
        src: [B, S]
        """
        self.eval()

        max_len = max_len or self.config.decoder.max_len
        src = src[:, :self.config.encoder.max_len]

        if beam_size == 1:
            return self._greedy_decode(src, max_len)
        else:
            return self._beam_search(src, max_len, beam_size)

    @torch.no_grad()
    def _greedy_decode(self, src, max_len):
        """
        Greedy decoding
        """
        src_mask = self.make_src_mask(src)
        enc = self.encoder(src, src_mask)

        B = src.size(0)
        device = src.device

        dec = torch.full((B, 1), self.vocab.bos_idx, device=device)
        outputs = []

        for _ in range(max_len):
            trg_mask = self.make_trg_mask(dec)
            out = self.decoder(dec, enc, trg_mask, src_mask)

            next_tok = out[:, -1].argmax(dim=-1, keepdim=True)
            outputs.append(next_tok)

            dec = torch.cat([dec, next_tok], dim=1)

            if (next_tok == self.vocab.eos_idx).all():
                break

        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def _beam_search(self, src, max_len, beam_size):
        """
        Simple beam search with length penalty.
        NOTE: batch_size = 1 only (as in paper inference).
        """
        alpha = getattr(self.config, "length_penalty", 2.0)

        src_mask = self.make_src_mask(src)
        enc = self.encoder(src, src_mask)

        assert src.size(0) == 1, "Beam search supports batch_size = 1 only"

        device = src.device
        vocab_size = self.vocab.vocab_size

        beams = torch.full((beam_size, 1), self.vocab.bos_idx, device=device)
        beam_scores = torch.zeros(beam_size, device=device)
        beam_scores[1:] = -1e9

        finished = []

        for step in range(max_len):
            enc_rep = enc.repeat(beam_size, 1, 1)
            src_mask_rep = src_mask.repeat(beam_size, 1, 1, 1)

            trg_mask = self.make_trg_mask(beams)
            out = self.decoder(beams, enc_rep, trg_mask, src_mask_rep)

            log_probs = torch.log_softmax(out[:, -1], dim=-1)
            scores = beam_scores.unsqueeze(1) + log_probs
            scores = scores.view(-1)

            top_scores, top_ids = scores.topk(beam_size)

            next_beams = []
            next_scores = []

            for score, idx in zip(top_scores, top_ids):
                beam_id = idx // vocab_size
                token_id = idx % vocab_size

                new_beam = torch.cat(
                    [beams[beam_id], token_id.view(1)], dim=0
                )

                if token_id.item() == self.vocab.eos_idx:
                    length = new_beam.size(0)
                    lp = ((5 + length) ** alpha) / ((5 + 1) ** alpha)
                    finished.append((new_beam, score / lp))
                else:
                    next_beams.append(new_beam)
                    next_scores.append(score)

                if len(next_beams) == beam_size:
                    break

            if len(next_beams) == 0:
                break

            beams = torch.stack(next_beams)
            beam_scores = torch.stack(next_scores)

        if len(finished) == 0:
            best_idx = beam_scores.argmax()
            return beams[best_idx][1:].unsqueeze(0)

        finished.sort(key=lambda x: x[1], reverse=True)
        return finished[0][0][1:].unsqueeze(0)


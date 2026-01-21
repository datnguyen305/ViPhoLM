import math
import torch
import torch.nn as nn
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab

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


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model)
        )

    def forward(self, x):
        return self.net(x)

class FullMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.d_head = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Tq, _ = q.size()
        Tk = k.size(1)

        Q = self.w_q(q).view(B, Tq, self.n_head, self.d_head).transpose(1, 2)
        K = self.w_k(k).view(B, Tk, self.n_head, self.d_head).transpose(1, 2)
        V = self.w_v(v).view(B, Tk, self.n_head, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                torch.finfo(scores.dtype).min
            )


        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)

        return self.w_o(out)

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
                scores = scores.masked_fill(mask_h == 0, 
                torch.finfo(scores.dtype).min)

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, Vh)

            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)

        return self.w_o(out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout):
        super().__init__()
        self.attn = FullMultiHeadAttention(d_model, n_head, dropout)
        self.norm1 = LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.drop(self.attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout, stride):
        super().__init__()
        self.self_attn = FullMultiHeadAttention(d_model, n_head, dropout)
        self.norm1 = LayerNorm(d_model)

        self.cross_attn = HeposMultiHeadAttention(d_model, n_head, stride, dropout)
        self.norm2 = LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm3 = LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc, trg_mask, src_mask):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, trg_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc, enc, src_mask)))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x

@META_ARCHITECTURE.register()
class HeposTransformerSummarizer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.d_model = config.d_model
        vocab_size = len(vocab)
        self.pad = vocab.pad_idx

        self.emb = nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=self.pad)
        self.pos = nn.Embedding(config.max_len, config.d_model)

        self.encoder = nn.ModuleList([
            EncoderLayer(
                config.d_model,
                config.ffn_hidden,
                config.n_head,
                config.drop_prob
            )
            for _ in range(config.n_layers)
        ])

        self.decoder = nn.ModuleList([
            DecoderLayer(
                config.d_model,
                config.ffn_hidden,
                config.n_head,
                config.drop_prob,
                config.stride
            )
            for _ in range(config.n_layers)
        ])

        self.fc = nn.Linear(config.d_model, vocab.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.pad,
            label_smoothing=0.1   
        )

    def make_src_mask(self, src):
        return (src != self.pad).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        T = trg.size(1)
        pad = (trg != self.pad).unsqueeze(1).unsqueeze(2)
        causal = torch.tril(torch.ones(T, T, device=trg.device)).bool()
        return pad & causal.unsqueeze(0).unsqueeze(1)

    def forward(self, src, trg):
        B, S = src.size()
        T = trg.size(1)

        pos_s = torch.arange(S, device=src.device)
        pos_t = torch.arange(T, device=trg.device)

        enc = self.emb(src) + self.pos(pos_s)
        dec = self.emb(trg) + self.pos(pos_t)

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        for layer in self.encoder:
            enc = layer(enc, src_mask)

        for layer in self.decoder:
            dec = layer(dec, enc, trg_mask, src_mask)

        out = self.fc(dec)
        loss = self.loss_fn(out.view(-1, out.size(-1)), trg.view(-1))
        return out, loss

    @torch.no_grad()
    def predict(
        self,
        src: torch.Tensor,
        max_len: int = None,
        beam_size: int = 1
    ):
        """
        src: [B, S]
        return: [B, T]
        """
        self.eval()

        device = src.device
        B = src.size(0)

        max_len = max_len or self.config.decoder.max_len

        if beam_size == 1:
            return self._greedy_decode(src, max_len)
        else:
            return self._beam_search(src, max_len, beam_size)
        
    @torch.no_grad()
    def _greedy_decode(self, src, max_len):
        device = src.device
        B, S = src.size()

        src_mask = self.make_src_mask(src)

        # ----- encode -----
        pos_s = torch.arange(S, device=device)
        enc = self.emb(src) + self.pos(pos_s)

        for layer in self.encoder:
            enc = layer(enc, src_mask)

        # ----- init decoder -----
        ys = torch.full(
            (B, 1),
            self.vocab.bos_idx,
            dtype=torch.long,
            device=device
        )

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            T = ys.size(1)
            pos_t = torch.arange(T, device=device)

            dec = self.emb(ys) + self.pos(pos_t)
            trg_mask = self.make_trg_mask(ys)

            for layer in self.decoder:
                dec = layer(dec, enc, trg_mask, src_mask)

            logits = self.fc(dec)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

            ys = torch.cat([ys, next_token], dim=1)
            finished |= (next_token.squeeze(1) == self.vocab.eos_idx)

            if finished.all():
                break

        return ys[:, 1:]   # bỏ BOS

    torch.no_grad()
    def _beam_search(self, src, max_len, beam_size):
        assert src.size(0) == 1, "Beam search supports batch_size = 1 only"

        device = src.device
        vocab_size = self.vocab.vocab_size
        alpha = getattr(self.config, "length_penalty", 2.0)

        src_mask = self.make_src_mask(src)
        S = src.size(1)

        # ----- encode -----
        pos_s = torch.arange(S, device=device)
        enc = self.emb(src) + self.pos(pos_s)

        for layer in self.encoder:
            enc = layer(enc, src_mask)

        beams = torch.full(
            (beam_size, 1),
            self.vocab.bos_idx,
            dtype=torch.long,
            device=device
        )

        beam_scores = torch.zeros(beam_size, device=device)
        beam_scores[1:] = -1e9

        finished = []

        for step in range(max_len):
            T = beams.size(1)
            pos_t = torch.arange(T, device=device)

            dec = self.emb(beams) + self.pos(pos_t)
            trg_mask = self.make_trg_mask(beams)

            enc_rep = enc.repeat(beam_size, 1, 1)
            src_mask_rep = src_mask.repeat(beam_size, 1, 1, 1)

            for layer in self.decoder:
                dec = layer(dec, enc_rep, trg_mask, src_mask_rep)

            logits = self.fc(dec)
            log_probs = torch.log_softmax(logits[:, -1], dim=-1)

            scores = beam_scores.unsqueeze(1) + log_probs
            scores = scores.view(-1)

            top_scores, top_ids = scores.topk(beam_size)

            next_beams = []
            next_scores = []

            for score, idx in zip(top_scores, top_ids):
                beam_id = idx // vocab_size
                token_id = idx % vocab_size

                new_beam = torch.cat(
                    [beams[beam_id], token_id.view(1)],
                    dim=0
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

            if not next_beams:
                break

            beams = torch.stack(next_beams)
            beam_scores = torch.stack(next_scores)

        if finished:
            finished.sort(key=lambda x: x[1], reverse=True)
            return finished[0][0][1:].unsqueeze(0)

        best = beam_scores.argmax()
        return beams[best][1:].unsqueeze(0)

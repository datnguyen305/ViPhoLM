import torch
import torch.nn as nn
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab
import copy

# =========================================================
# Pointer Generator (Eq 9–10)
# =========================================================

class PointerGenerator(nn.Module):
    def __init__(self, enc_dim, hidden_dim, emb_dim):
        super().__init__()
        self.w_c = nn.Linear(enc_dim, 1, bias=False)
        self.w_s = nn.Linear(hidden_dim, 1, bias=False)
        self.w_y = nn.Linear(emb_dim, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        vocab_dist,     # (B, V)
        attn_dist,      # (B, T)
        context,        # (B, enc_dim)
        state,          # (B, hidden_dim)
        emb,            # (B, emb_dim)
        src_ids,        # (B, T)
        vocab_size
    ):
        p_gen = torch.sigmoid(
            self.w_c(context) +
            self.w_s(state) +
            self.w_y(emb) +
            self.b
        )  # (B,1)

        p_vocab = p_gen * vocab_dist
        p_copy = (1 - p_gen) * attn_dist

        B, T = src_ids.size()
        # final_dist chỉ có vocab_size
        final_dist = p_vocab.clone()

        # Chỉ copy các token hợp lệ trong vocab
        copy_mask = src_ids < vocab_size
        safe_src_ids = src_ids.clamp(max=vocab_size - 1)

        final_dist.scatter_add_(
            1,
            safe_src_ids,
            p_copy * copy_mask.float()
        )

        return final_dist


# =========================================================
# Attention
# =========================================================

class BahdanauAttention(nn.Module):
    def __init__(self, query_dim, key_dim, attn_dim):
        super().__init__()
        self.Wq = nn.Linear(query_dim, attn_dim, bias=False)
        self.Wk = nn.Linear(key_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, keys, mask=None):
        q = self.Wq(query).unsqueeze(1)
        k = self.Wk(keys)
        e = self.v(torch.tanh(q + k)).squeeze(-1)

        if mask is not None:
            e = e.masked_fill(mask == 0, torch.finfo(e.dtype).min)


        a = F.softmax(e, dim=-1)
        c = torch.bmm(a.unsqueeze(1), keys).squeeze(1)
        return c, a


# =========================================================
# Encoder
# =========================================================

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        outputs, _ = self.lstm(emb)
        return outputs   # (B, T, 2H)


# =========================================================
# Backward Decoder (NO pointer in training, pointer in step)
# =========================================================

class BackwardDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, enc_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim + enc_dim, hid_dim, batch_first=True)
        self.attn = BahdanauAttention(hid_dim, enc_dim, hid_dim)
        self.proj = nn.Linear(hid_dim, vocab_size)

    def step(self, y, state, enc_out, src_ids, src_mask):
        emb = self.embedding(y)
        query = state[0].squeeze(0) if state else emb.new_zeros(emb.size(0), self.lstm.hidden_size)
        ctx, attn = self.attn(query, enc_out, src_mask)

        x = torch.cat([emb, ctx], dim=-1).unsqueeze(1)
        out, state = self.lstm(x, state)

        vocab_dist = F.softmax(self.proj(out.squeeze(1)), dim=-1)
        return vocab_dist, state, out.squeeze(1)

    def forward(self, tgt_rev, enc_out, src_mask):
        B, L = tgt_rev.size()
        emb = self.embedding(tgt_rev)

        h, c = None, None
        logits = []
        hidden_states = []

        for t in range(L):
            query = h.squeeze(0) if h is not None else torch.zeros(
                B, self.lstm.hidden_size, device=tgt_rev.device
            )
            ctx, _ = self.attn(query, enc_out, src_mask)
            x = torch.cat([emb[:, t], ctx], dim=-1).unsqueeze(1)
            out, (h, c) = self.lstm(x, (h, c) if h is not None else None)

            hidden_states.append(out.squeeze(1))
            logits.append(self.proj(out.squeeze(1)))

        return torch.stack(logits, dim=1), torch.stack(hidden_states, dim=1)


# =========================================================
# Forward Decoder (WITH pointer)
# =========================================================

class ForwardDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, enc_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim + enc_dim + hid_dim, hid_dim, batch_first=True)

        self.attn_enc = BahdanauAttention(hid_dim, enc_dim, hid_dim)
        self.attn_bwd = BahdanauAttention(hid_dim, hid_dim, hid_dim)

        self.proj = nn.Linear(hid_dim, vocab_size)
        self.pointer = PointerGenerator(enc_dim, hid_dim, emb_dim)

    def step(self, y, state, enc_out, bwd_states, src_ids, src_mask):
        emb = self.embedding(y)
        query = state[0].squeeze(0) if state else emb.new_zeros(emb.size(0), self.lstm.hidden_size)

        c_enc, attn_enc = self.attn_enc(query, enc_out, src_mask)
        c_bwd, _ = self.attn_bwd(query, bwd_states)

        x = torch.cat([emb, c_enc, c_bwd], dim=-1).unsqueeze(1)
        out, state = self.lstm(x, state)

        vocab_dist = F.softmax(self.proj(out.squeeze(1)), dim=-1)
        final_dist = self.pointer(
            vocab_dist,
            attn_enc,
            c_enc,
            out.squeeze(1),
            emb,
            src_ids,
            vocab_dist.size(-1)
        )
        return final_dist, state


# =========================================================
# BiSum Model
# =========================================================

@META_ARCHITECTURE.register()
class BiSumModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.pad_idx = vocab.pad_idx
        self.d_model = config.d_model
        self.max_len = config.max_len
        
        self.encoder = BiLSTMEncoder(
            vocab.vocab_size,
            config.emb_dim,
            self.d_model // 2,
            self.pad_idx
        )

        self.backward_decoder = BackwardDecoder(
            vocab.vocab_size,
            config.emb_dim,
            self.d_model // 2,
            self.d_model,
            self.pad_idx
        )

        self.forward_decoder = ForwardDecoder(
            vocab.vocab_size,
            config.emb_dim,
            self.d_model // 2,
            self.d_model,
            self.pad_idx
        )

        self.lambda_mix = config.lambda_mix

    # ========================= TRAIN =========================

    def forward(self, input_ids, labels):
        src_mask = (input_ids != self.pad_idx).bool()
        enc_out = self.encoder(input_ids)

        tgt_rev = torch.flip(labels, dims=[1])
        bwd_logits, bwd_states = self.backward_decoder(tgt_rev, enc_out, src_mask)
        bwd_states = bwd_states.detach()

        loss_bwd = F.cross_entropy(
            bwd_logits.view(-1, bwd_logits.size(-1)),
            tgt_rev.view(-1),
            ignore_index=self.pad_idx
        )

        loss_fwd = self.forward_loss(labels, enc_out, bwd_states, input_ids, src_mask)

        loss = self.lambda_mix * loss_fwd + (1 - self.lambda_mix) * loss_bwd
        return None, loss

    def forward_loss(self, labels, enc_out, bwd_states, src_ids, src_mask):
        """
        Teacher-forcing forward decoder loss with pointer-generator
        """
        B, L = labels.size()
        state = None
        losses = []

        for t in range(L - 1):
            # y_t: input token at time t
            y_t = labels[:, t]

            # one decoding step
            dist, state = self.forward_decoder.step(
                y_t,
                state,
                enc_out,
                bwd_states,
                src_ids,
                src_mask
            )

            # target is y_{t+1}
            loss_t = F.nll_loss(
                torch.log(dist + 1e-12),
                labels[:, t + 1],
                ignore_index=self.pad_idx
            )
            losses.append(loss_t)

        return torch.stack(losses).mean()



    # ========================= INFER =========================

    def backward_beam_search(self, enc_out, src_ids, src_mask, max_len=50, beam_size=2):
        beams = [([self.vocab.eos_idx], 0.0, None, [])]

        for _ in range(max_len):
            new_beams = []
            for tokens, score, state, states in beams:
                y = torch.tensor([tokens[-1]], device=enc_out.device)
                dist, new_state, h = self.backward_decoder.step(y, state, enc_out, src_ids, src_mask)
                logp, topk = torch.topk(torch.log(dist), beam_size)
                for i in range(beam_size):
                    new_beams.append((
                        tokens + [topk[i].item()],
                        score + logp[i].item(),
                        (new_state[0].clone(), new_state[1].clone()),
                        states + [h]
                    ))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        return torch.stack(beams[0][3], dim=1)

    def predict(self, input_ids, max_len=None):
        max_len = max_len if max_len is not None else self.max_len
        src_mask = (input_ids != self.pad_idx).bool()
        enc_out = self.encoder(input_ids)

        with torch.no_grad():
            bwd_states = self.backward_beam_search(enc_out, input_ids, src_mask)

        y = torch.full((1, 1), self.vocab.bos_idx, device=input_ids.device)
        state = None
        outputs = []

        for _ in range(max_len):
            dist, state = self.forward_decoder.step(
                y[:, -1], state,
                enc_out, bwd_states,
                input_ids, src_mask
            )
            token = dist.argmax(-1)
            if token.item() == self.vocab.eos_idx:
                break
            outputs.append(token)
            y = torch.cat([y, token.unsqueeze(1)], dim=1)

        return torch.stack(outputs, dim=1)

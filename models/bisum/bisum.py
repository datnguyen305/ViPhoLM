import torch
import torch.nn as nn
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab


class PointerGenerator(nn.Module):
    def __init__(self, enc_dim, hid_dim, emb_dim):
        super().__init__()
        self.w_c = nn.Linear(enc_dim, 1, bias=False)
        self.w_s = nn.Linear(hid_dim, 1, bias=False)
        self.w_y = nn.Linear(emb_dim, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        vocab_dist,     # (B, V)
        attn_dist,      # (B, T)
        context,        # (B, enc_dim)
        state,          # (B, hid_dim)
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
        p_copy = (1.0 - p_gen) * attn_dist

        final_dist = p_vocab.clone()

        copy_mask = src_ids < vocab_size
        safe_src = src_ids.clamp(max=vocab_size - 1)

        final_dist.scatter_add_(
            1,
            safe_src,
            p_copy * copy_mask.float()
        )
        return final_dist


class BahdanauAttention(nn.Module):
    def __init__(self, q_dim, k_dim, attn_dim):
        super().__init__()
        self.Wq = nn.Linear(q_dim, attn_dim, bias=False)
        self.Wk = nn.Linear(k_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, keys, mask=None):
        q = self.Wq(query).unsqueeze(1)      # (B,1,D)
        k = self.Wk(keys)                    # (B,T,D)
        e = self.v(torch.tanh(q + k)).squeeze(-1)  # (B,T)

        if mask is not None:
            e = e.masked_fill(mask == 0, torch.finfo(e.dtype).min)

        a = F.softmax(e, dim=-1)
        c = torch.bmm(a.unsqueeze(1), keys).squeeze(1)
        return c, a


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
        return outputs    # (B,T,2H)


class BackwardDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, enc_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim + enc_dim, hid_dim, batch_first=True)
        self.attn = BahdanauAttention(hid_dim, enc_dim, hid_dim)
        self.proj = nn.Linear(hid_dim, vocab_size)

    def forward(self, tgt_rev, enc_out, src_mask):
        B, L = tgt_rev.size()
        emb = self.embedding(tgt_rev)

        h, c = None, None
        logits, states = [], []

        for t in range(L):
            query = h.squeeze(0) if h is not None else emb.new_zeros(B, self.lstm.hidden_size)
            ctx, _ = self.attn(query, enc_out, src_mask)

            x = torch.cat([emb[:, t], ctx], dim=-1).unsqueeze(1)
            out, (h, c) = self.lstm(x, (h, c) if h is not None else None)

            logits.append(self.proj(out.squeeze(1)))
            states.append(out.squeeze(1))

        return torch.stack(logits, 1), torch.stack(states, 1)

    def step(self, y, state, enc_out, src_mask):
        emb = self.embedding(y)
        query = state[0].squeeze(0) if state else emb.new_zeros(emb.size(0), self.lstm.hidden_size)
        ctx, attn = self.attn(query, enc_out, src_mask)

        x = torch.cat([emb, ctx], dim=-1).unsqueeze(1)
        out, state = self.lstm(x, state)
        dist = F.softmax(self.proj(out.squeeze(1)), dim=-1)
        return dist, state, out.squeeze(1)



class ForwardDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, enc_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim + enc_dim + hid_dim, hid_dim, batch_first=True)

        self.attn_enc = BahdanauAttention(hid_dim, enc_dim, hid_dim)
        self.attn_bwd = BahdanauAttention(hid_dim, hid_dim, hid_dim)

        self.proj = nn.Linear(hid_dim, vocab_size)
        self.pointer = PointerGenerator(enc_dim, hid_dim, emb_dim)

    def forward_train(self, tgt, enc_out, bwd_states, src_ids, src_mask):
        emb = self.embedding(tgt)             # (B,L,E)
        B, L, _ = emb.size()

        h0 = emb.new_zeros(1, B, self.lstm.hidden_size)
        c0 = emb.new_zeros(1, B, self.lstm.hidden_size)

        q0 = h0.squeeze(0)
        c_enc, attn_enc = self.attn_enc(q0, enc_out, src_mask)
        c_bwd, _ = self.attn_bwd(q0, bwd_states)

        c_enc = c_enc.unsqueeze(1).expand(-1, L, -1)
        c_bwd = c_bwd.unsqueeze(1).expand(-1, L, -1)

        x = torch.cat([emb, c_enc, c_bwd], dim=-1)
        out, _ = self.lstm(x, (h0, c0))

        vocab_dist = F.softmax(self.proj(out), dim=-1)

        final = []
        for t in range(L):
            final.append(
                self.pointer(
                    vocab_dist[:, t],
                    attn_enc,
                    c_enc[:, t],
                    out[:, t],
                    emb[:, t],
                    src_ids,
                    vocab_dist.size(-1)
                )
            )
        return torch.stack(final, dim=1)


    def step(self, y, state, enc_out, bwd_states, src_ids, src_mask):
        emb = self.embedding(y)
        query = state[0].squeeze(0) if state else emb.new_zeros(emb.size(0), self.lstm.hidden_size)

        c_enc, attn_enc = self.attn_enc(query, enc_out, src_mask)
        c_bwd, _ = self.attn_bwd(query, bwd_states)

        x = torch.cat([emb, c_enc, c_bwd], dim=-1).unsqueeze(1)
        out, state = self.lstm(x, state)

        vocab_dist = F.softmax(self.proj(out.squeeze(1)), dim=-1)
        final = self.pointer(
            vocab_dist, attn_enc, c_enc,
            out.squeeze(1), emb, src_ids, vocab_dist.size(-1)
        )
        return final, state


@META_ARCHITECTURE.register()
class BiSumModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.pad_idx = vocab.pad_idx
        self.max_len = config.max_len
        self.lambda_mix = config.lambda_mix
        self.d_model = config.d_model

        self.encoder = BiLSTMEncoder(
            vocab.vocab_size,
            config.emb_dim,
            config.d_model // 2,
            self.pad_idx
        )

        self.backward_decoder = BackwardDecoder(
            vocab.vocab_size,
            config.emb_dim,
            config.d_model // 2,
            config.d_model,
            self.pad_idx
        )

        self.forward_decoder = ForwardDecoder(
            vocab.vocab_size,
            config.emb_dim,
            config.d_model // 2,
            config.d_model,
            self.pad_idx
        )


    def forward(self, input_ids, labels):
        src_mask = (input_ids != self.pad_idx)
        enc_out = self.encoder(input_ids)

        tgt_rev = torch.flip(labels, dims=[1])

        with torch.no_grad():
            bwd_logits, bwd_states = self.backward_decoder(tgt_rev, enc_out, src_mask)

        loss_bwd = F.cross_entropy(
            bwd_logits.reshape(-1, bwd_logits.size(-1)),
            tgt_rev.reshape(-1),
            ignore_index=self.pad_idx
        )

        dist = self.forward_decoder.forward_train(
            labels[:, :-1],
            enc_out,
            bwd_states,
            input_ids,
            src_mask
        )

        loss_fwd = F.nll_loss(
            torch.log(dist + 1e-12).reshape(-1, dist.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=self.pad_idx
        )

        loss = self.lambda_mix * loss_fwd + (1.0 - self.lambda_mix) * loss_bwd
        return None, loss


    def backward_greedy(self, enc_out, src_ids, src_mask):
        """
        Greedy decoding for backward decoder.
        Returns hidden states (1, L, H)
        """
        B = enc_out.size(0)
        assert B == 1, "Backward greedy decoding only supports batch_size=1"
    
        y = torch.full(
            (1,),
            self.vocab.eos_idx,
            device=enc_out.device,
            dtype=torch.long
        )
    
        state = None
        hidden_states = []
    
        for _ in range(self.max_len):
            dist, state, h = self.backward_decoder.step(
                y,
                state,
                enc_out,
                src_mask
            )
    
            y = dist.argmax(dim=-1)
    
            if y.item() == self.vocab.bos_idx:
                break
    
            hidden_states.append(h)
    
        if len(hidden_states) == 0:
            # fallback tránh crash
            hidden_states.append(
                enc_out.new_zeros(1, self.forward_decoder.lstm.hidden_size)
            )
    
        return torch.stack(hidden_states, dim=1)


    def predict(self, input_ids, max_len=None):
        max_len = max_len or self.max_len
        src_mask = (input_ids != self.pad_idx)
        enc_out = self.encoder(input_ids)

        with torch.no_grad():
            bwd_states = self.backward_greedy(enc_out, input_ids, src_mask)

        y = torch.full((1, 1), self.vocab.bos_idx, device=input_ids.device)
        state, outputs = None, []

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


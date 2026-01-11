import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class TemporalCNN(nn.Module):
    def __init__(self, d_model, out_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, out_dim, kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x):
        # x: (B, L, D)
        x = x.transpose(1, 2)          # (B, D, L)
        h = F.relu(self.conv(x))
        h = torch.max(h, dim=2)[0]     # (B, out_dim)
        return h


class EntityAwareContentSelector(nn.Module):
    def __init__(self, vocab_size, d_model=128, hidden_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Encoders
        self.entity_encoder = TemporalCNN(d_model, hidden_dim)
        self.sent_encoder = TemporalCNN(d_model, hidden_dim)
        self.sent_bilstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        # Decoder
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Attention
        self.Ws = nn.Linear(hidden_dim, hidden_dim)
        self.We = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

        self.Wc = nn.Linear(hidden_dim, hidden_dim)
        self.vc = nn.Linear(hidden_dim, 1)

        # END token score
        self.end_linear = nn.Linear(hidden_dim, 1)

    def encode_entities(self, entities, device):
        """
        entities: List[B] of List[E] of List[int]
        return: Tensor (B, E, H)
        """
        batch_reps = []

        for ent_list in entities:  # loop over batch
            ent_reps = []

            for e in ent_list:  # each entity cluster
                if isinstance(e, torch.Tensor):
                    e = e.to(device)
                else:
                    e = torch.tensor(e, dtype=torch.long, device=device)

                emb = self.embedding(e.unsqueeze(0))   # (1, L, D)
                rep = self.entity_encoder(emb)         # (1, H)
                ent_reps.append(rep.squeeze(0))

            if len(ent_reps) == 0:
                # edge case: no entity
                ent_reps.append(torch.zeros(
                    self.entity_encoder.conv.out_channels,
                    device=device
                ))

            ent_reps = torch.stack(ent_reps, dim=0)  # (E, H)
            batch_reps.append(ent_reps)

        return torch.stack(batch_reps, dim=0)        # (B, E, H)


    def encode_sentences(self, sentences):
        # sentences: (B, N, L)
        B, N, L = sentences.size()
        flat = sentences.view(B * N, L)
        emb = self.embedding(flat)
        h = self.sent_encoder(emb)
        h = h.view(B, N, -1)
        h, _ = self.sent_bilstm(h)
        return h

    def forward(self, sentences, entities, max_steps=5):
        device = sentences.device
        B, N, _ = sentences.size()

        sent_h = self.encode_sentences(sentences)
        ent_h = self.encode_entities(entities, device)


        h_t = torch.zeros(1, B, sent_h.size(-1), device=device)
        c_t = torch.zeros_like(h_t)

        mask = torch.ones(B, N, device=device)
        selected = []

        dec_input = torch.zeros(B, 1, sent_h.size(-1), device=device)

        for _ in range(max_steps):
            out, (h_t, c_t) = self.decoder(dec_input, (h_t, c_t))
            s_t = out.squeeze(1)

            # Entity attention
            e_score = self.v(torch.tanh(
                self.We(ent_h) + self.Ws(s_t).unsqueeze(1)
            )).squeeze(-1)
            e_attn = F.softmax(e_score, dim=1)
            c_e = torch.sum(e_attn.unsqueeze(-1) * ent_h, dim=1)

            # Sentence attention
            score = self.vc(torch.tanh(
                self.Wc(sent_h) + self.Ws(s_t).unsqueeze(1) + c_e.unsqueeze(1)
            )).squeeze(-1)

            score = score.masked_fill(mask == 0, -1e9)
            attn = F.softmax(score, dim=1)

            idx = torch.argmax(attn, dim=1)
            selected.append(idx)

            mask[torch.arange(B), idx] = 0
            dec_input = sent_h[torch.arange(B), idx].unsqueeze(1)

        return selected

class PointerGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            embed_dim + hidden_dim * 2,
            hidden_dim,
            batch_first=True
        )

        self.attn = nn.Linear(hidden_dim * 4, 1)  # 1024
        self.dec_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.vocab_proj = nn.Linear(hidden_dim * 3, vocab_size)
        self.p_gen = nn.Linear(hidden_dim * 3 + embed_dim, 1)

    def forward(self, source, target=None, max_len=100):
        B, L = source.size()
        emb = self.embedding(source)

        enc_out, (h, c) = self.encoder(emb)

        h = h[-2:].transpose(0, 1).contiguous().view(1, B, -1)
        c = c[-2:].transpose(0, 1).contiguous().view(1, B, -1)

        h = h[:, :, :self.hidden_dim].contiguous()
        c = c[:, :, :self.hidden_dim].contiguous()


        dec_input = torch.zeros(B, 1, dtype=torch.long, device=source.device)
        outputs = []

        T = target.size(1) if target is not None else max_len

        for t in range(T):
            emb_t = self.embedding(dec_input)
            h_t = h.squeeze(0)

            dec_proj = self.dec_proj(h_t)                 # (B, 512)
            dec_proj = dec_proj.unsqueeze(1)              # (B, 1, 512)
            dec_proj = dec_proj.expand_as(enc_out)         # (B, L, 512)

            attn_score = self.attn(
                torch.cat([enc_out, dec_proj], dim=2)
            ).squeeze(-1)


            attn_w = F.softmax(attn_score, dim=1)
            context = torch.sum(attn_w.unsqueeze(-1) * enc_out, dim=1)

            dec_in = torch.cat([emb_t, context.unsqueeze(1)], dim=2)
            out, (h, c) = self.decoder(dec_in, (h, c))

            gen_feat = torch.cat([out.squeeze(1), context], dim=1)
            vocab_dist = F.softmax(self.vocab_proj(gen_feat), dim=1)

            p_gen = torch.sigmoid(
                self.p_gen(torch.cat([gen_feat, emb_t.squeeze(1)], dim=1))
            )

            copy_dist = torch.zeros_like(vocab_dist)
            copy_dist.scatter_add_(1, source, attn_w)

            final = p_gen * vocab_dist + (1 - p_gen) * copy_dist
            outputs.append(final)

            dec_input = (
                target[:, t].unsqueeze(1)
                if target is not None else
                torch.argmax(final, dim=1).unsqueeze(1)
            )

        return torch.stack(outputs, dim=1)

@META_ARCHITECTURE.register()
class SENECA_Baseline(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        vocab_size = len(vocab)

        d_model = config.d_model
        hidden_dim = config.hidden_dim

        self.selector = EntityAwareContentSelector(
            vocab_size=vocab_size,
            d_model=d_model,
            hidden_dim=hidden_dim
        )

        self.abstractor = PointerGenerator(
            vocab_size=vocab_size,
            embed_dim=d_model,
            hidden_dim=hidden_dim
        )

        self.max_extract = config.max_extract


    def forward(
        self,
        sentences,
        entities,
        target=None,
        return_coverage=False
    ):
        """
        sentences: (B, N, L)
        entities:  List[List[int]]
        target:    (B, T) or None
        """

        selected_indices = self.selector(
            sentences,
            entities,
            max_steps=self.max_extract
        )

        B = sentences.size(0)
        selected_sources = []

        for b in range(B):
            sent_list = []
            for idx in selected_indices:
                sent_list.append(sentences[b, idx[b]])
            selected_sources.append(torch.cat(sent_list, dim=0))

        selected_sources = torch.stack(selected_sources)  # (B, K*L)

        outputs = self.abstractor(
            selected_sources,
            target=target
        )

        # ---- coverage compatibility ----
        if return_coverage:
            cov_loss = torch.tensor(0.0, device=sentences.device)
            return outputs, selected_indices, None, cov_loss

        return outputs, selected_indices, None



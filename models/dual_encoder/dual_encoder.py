import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


class Attention(nn.Module):
    def __init__(self, n_hidden_enc, n_hidden_dec):
        super().__init__()
        self.W = nn.Linear(n_hidden_enc + n_hidden_dec, n_hidden_dec, bias=False)
        self.V = nn.Parameter(torch.rand(n_hidden_dec))

    def forward(self, hidden_dec, encoder_outputs):
        batch, seq_len, _ = encoder_outputs.size()

        hidden_dec = hidden_dec.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.W(torch.cat([hidden_dec, encoder_outputs], dim=2)))

        energy = energy.permute(0, 2, 1)
        V = self.V.repeat(batch, 1).unsqueeze(1)

        scores = torch.bmm(V, energy).squeeze(1)
        attn = F.softmax(scores, dim=1)

        context = torch.bmm(attn.unsqueeze(1), encoder_outputs).squeeze(1)
        return attn, context

class PrimaryEncoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)

        self.gru = nn.GRU(
            config.hidden_size,
            config.hidden_size // 2,
            bidirectional=True,
            num_layers=config.layer_dim,
            batch_first=True,
            dropout=config.dropout if config.layer_dim > 1 else 0.0
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        emb = self.dropout(self.embedding(input))
        outputs, h = self.gru(emb)

        # h: [layers*2, B, H/2] → [layers, B, H]
        h = torch.cat([h[0::2], h[1::2]], dim=-1)

        return outputs, h

    
class ImportanceScorer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size * 3, 1)

    def forward(self, h_p, C_p, C_d):
        B, T, H = h_p.size()
        Cp = C_p.unsqueeze(1).expand(-1, T, -1)
        Cd = C_d.unsqueeze(1).expand(-1, T, -1)

        score = self.fc(torch.cat([h_p, Cp, Cd], dim=-1)).squeeze(-1)
        alpha = torch.sigmoid(score)  
        return alpha

    
class SecondaryEncoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)

        self.gru = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            num_layers=config.layer_dim,
            batch_first=True,
            dropout=config.dropout if config.layer_dim > 1 else 0.0
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input, alpha):
        emb = self.dropout(self.embedding(input))
        emb = emb * alpha.unsqueeze(-1)

        outputs, states = self.gru(emb)
        return outputs, states

    
class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.gru = nn.GRU(
            config.hidden_size * 2,
            config.hidden_size,
            batch_first=True
        )
        
        self.attention = Attention(config.hidden_size, config.hidden_size)
        self.fc_out = nn.Linear(config.hidden_size * 3, vocab.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward_step(self, input, hidden, encoder_outputs):
        h_dec = hidden[-1]

        _, context = self.attention(h_dec, encoder_outputs)
        emb = self.dropout(self.embedding(input))

        gru_input = torch.cat([emb, context.unsqueeze(1)], dim=2)
        out, hidden = self.gru(gru_input, hidden)

        combined = torch.cat(
            [out.squeeze(1), context, emb.squeeze(1)], dim=1
        )

        logits = self.fc_out(combined).unsqueeze(1)
        return logits, hidden
    
@META_ARCHITECTURE.register()
class DualEncodingGRU(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.K = config.stage_decode_k   # ✅ FIX
        self.MAX_LEN = vocab.max_sentence_length + 2

        self.primary_encoder = PrimaryEncoder(config.encoder, vocab)
        self.secondary_encoder = SecondaryEncoder(config.secondary_encoder, vocab)

        self.importance = ImportanceScorer(config.encoder.hidden_size)
        self.decoder = Decoder(config.decoder, vocab)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=config.label_smoothing)


    def forward(self, x, y):
        h_p, h_p_last = self.primary_encoder(x)
        C_p = h_p.mean(dim=1)

        hidden = h_p_last
        decoder_input = torch.full(
            (x.size(0), 1),
            self.vocab.bos_idx,
            dtype=torch.long,
            device=x.device
        )

        outputs = []
        C_d = torch.zeros_like(C_p)

        for t in range(y.size(1)):
            if t % self.K == 0:
                alpha = self.importance(h_p, C_p, C_d)
                h_s, _ = self.secondary_encoder(x, alpha)
            else:
                h_s = h_p

            out, hidden = self.decoder.forward_step(
                decoder_input, hidden, h_s
            )

            outputs.append(out)
            decoder_input = y[:, t].unsqueeze(1)

            if (t + 1) % self.K == 0:
                C_d = torch.cat(outputs, dim=1).mean(dim=1)

        logits = torch.cat(outputs, dim=1)
        loss = self.loss_fn(
            logits.reshape(-1, self.vocab.vocab_size),
            y.reshape(-1)
        )

        return logits, loss


    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """
        Greedy decoding for inference
        x: [batch, src_len]
        return: [batch, decoded_len]
        """

        device = x.device
        batch_size = x.size(0)

        # ---------- Encode source ----------
        h_p, h_p_last = self.primary_encoder(x)
        C_p = h_p.mean(dim=1)

        hidden = h_p_last
        decoder_input = torch.full(
            (batch_size, 1),
            self.vocab.bos_idx,
            dtype=torch.long,
            device=device
        )

        outputs = []
        C_d = torch.zeros_like(C_p)

        for t in range(self.MAX_LEN):

            # -------- Stage-level re-encoding --------
            if t % self.K == 0:
                alpha = self.importance(h_p, C_p, C_d)
                h_s, _ = self.secondary_encoder(x, alpha)
            else:
                h_s = h_p

            # -------- Decode one step --------
            logits, hidden = self.decoder.forward_step(
                decoder_input, hidden, h_s
            )

            next_token = logits.argmax(dim=-1)  # [B, 1]
            outputs.append(next_token)

            decoder_input = next_token

            # -------- Update decoded content --------
            if (t + 1) % self.K == 0:
                C_d = torch.cat(outputs, dim=1).mean(dim=1)

            # -------- Early stop --------
            if (next_token == self.vocab.eos_idx).all():
                break

        outputs = torch.cat(outputs, dim=1)
        return outputs

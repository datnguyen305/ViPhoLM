import math
import torch
import torch.nn as nn
from vocabs.hierarchy_vocab import Hierarchy_Vocab
from builders.model_builder import META_ARCHITECTURE


class SentenceAwareSelfAttention(nn.Module):
    """
    Input  : (B, N_sent, S_token, D)
    Output : (B, N_sent, S_token, D)
    """

    def __init__(self, hidden_size, n_head):
        super().__init__()
        assert hidden_size % n_head == 0

        self.hidden_size = hidden_size
        self.n_head = n_head
        self.d_k = hidden_size // n_head

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # sentence semantic query (per head)
        self.sent_query = nn.Parameter(
            torch.randn(n_head, self.d_k)
        )

        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask, return_sentence_importance=False):
        """
        x    : (B, N, S, D)
        mask : (B, N, S)
        """
        B, N, S, D = x.size()

        Q = self.q_proj(x).view(B, N, S, self.n_head, self.d_k)
        K = self.k_proj(x).view(B, N, S, self.n_head, self.d_k)
        V = self.v_proj(x).view(B, N, S, self.n_head, self.d_k)

        Q = Q.permute(0, 3, 1, 2, 4)
        K = K.permute(0, 3, 1, 2, 4)
        V = V.permute(0, 3, 1, 2, 4)

        sent_q = self.sent_query.view(1, self.n_head, 1, 1, self.d_k)
        sent_scores = torch.matmul(
            sent_q, K.transpose(-2, -1)
        ) / math.sqrt(self.d_k)

        sent_scores = sent_scores.masked_fill(
            mask.unsqueeze(1).unsqueeze(3) == 0,
            -1e9
        )

        sent_attn = torch.softmax(sent_scores, dim=-1)
        sent_repr = torch.matmul(sent_attn, V)  # (B,H,N,1,d_k)

        token_scores = torch.matmul(
            Q, sent_repr.transpose(-2, -1)
        ) / math.sqrt(self.d_k)

        token_attn = torch.softmax(token_scores, dim=-1)
        out = token_attn * sent_repr

        out = out.permute(0, 2, 3, 1, 4).reshape(B, N, S, D)
        out = self.out_proj(out)

        if return_sentence_importance:
            # (B, N) – average over heads & tokens
            sent_importance = sent_attn.mean(dim=(1, 3, 4))
            return out, sent_importance

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_head):
        super().__init__()
        assert hidden_size % n_head == 0
        self.n_head = n_head
        self.d_k = hidden_size // n_head
        self.sent_bias_scale = nn.Parameter(torch.tensor(0.1))


        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, mask=None, causal_mask=None, sentence_bias=None):
        B, T, D = q.size()
        S = k.size(1)

        Q = self.q_proj(q).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        K = self.k_proj(k).view(B, S, self.n_head, self.d_k).transpose(1, 2)
        V = self.v_proj(v).view(B, S, self.n_head, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if sentence_bias is not None:
            # sentence_bias: (B, S_enc)
            scores = scores + self.sent_bias_scale * sentence_bias.unsqueeze(1).unsqueeze(1)


        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_size)
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SentenceAwareSelfAttention(
            config.hidden_size, config.n_head
        )
        self.ffn = FeedForward(
            config.hidden_size, config.ffn_hidden, config.drop_prob
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.drop_prob)

    def forward(self, x, mask, collect_sentence_importance=False):
        if collect_sentence_importance:
            attn_out, sent_imp = self.attn(
                self.norm1(x), mask, return_sentence_importance=True
            )
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x, sent_imp
        else:
            x = x + self.dropout(self.attn(self.norm1(x), mask))
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            config.hidden_size, config.n_head
        )
        self.cross_attn = MultiHeadAttention(
            config.hidden_size, config.n_head
        )
        self.ffn = FeedForward(
            config.hidden_size, config.ffn_hidden, config.drop_prob
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.drop_prob)

    def forward(self, x, memory, trg_mask, trg_causal_mask, src_mask, sentence_bias=None):
        x = x + self.dropout(
            self.self_attn(self.norm1(x), x, x, trg_mask, trg_causal_mask)
        )
        x = x + self.dropout(
            self.cross_attn(
                self.norm2(x),
                memory,
                memory,
                src_mask,
                sentence_bias=sentence_bias
            )
        )

        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


@META_ARCHITECTURE.register()
class HierarchicalTransformer(nn.Module):
    def __init__(self, config, vocab: Hierarchy_Vocab):
        super().__init__()
        self.d_model = config.d_model
        self.embed = nn.Embedding(
            vocab.vocab_size, self.d_model,
            padding_idx=vocab.pad_idx
        )
        self.pe = PositionalEncoding(self.d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.encoder)
            for _ in range(config.encoder.n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.decoder)
            for _ in range(config.decoder.n_layers)
        ])

        self.fc_out = nn.Linear(self.d_model, vocab.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.vocab = vocab

    def forward(self, src, trg):
        """
        src: (B, N_sent, S_token)
        trg: (B, T)
        """
        B, N, S = src.size()

        trg_input = trg[:, :-1]
        trg_label = trg[:, 1:]

        src_emb = self.embed(src)
        src_mask = (src != self.vocab.pad_idx)

        memory = src_emb
        sentence_importance = None
        for layer in self.encoder_layers:
            memory, sentence_importance = layer(
                memory, src_mask, collect_sentence_importance=True
            )
        
        #flatten memory
        B, N, S, D = memory.size()
        memory = memory.contiguous().view(B, N * S, D)

        sentence_ids = torch.arange(N, device=memory.device).repeat_interleave(S)
        sentence_bias = sentence_importance[:, sentence_ids]  # (B, N*S)

        src_mask = src_mask.view(B, N * S).unsqueeze(1).unsqueeze(2)

        dec_emb = self.pe(self.embed(trg_input))

        trg_mask = (trg_input != self.vocab.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_causal_mask = torch.tril(
            torch.ones(trg_input.size(1), trg_input.size(1), device=trg.device)
        ).bool()

        out = dec_emb
        for layer in self.decoder_layers:
            out = layer(
                out,
                memory,
                trg_mask,
                trg_causal_mask,
                src_mask,
                sentence_bias
            )

        logits = self.fc_out(out)
        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            trg_label.reshape(-1)
        )

        return logits, loss

    @torch.no_grad()
    def predict(self, src, max_len=200):
        device = src.device
        B, N, S = src.size()

        src_emb = self.embed(src)
        src_mask = (src != self.vocab.pad_idx)

        memory = src_emb
        sentence_importance = None
        for layer in self.encoder_layers:
            memory, sentence_importance = layer(
                memory, src_mask, collect_sentence_importance=True
            )

        B, N, S, D = memory.size()
        memory = memory.contiguous().view(B, N * S, D)
        src_mask = src_mask.view(B, N * S).unsqueeze(1).unsqueeze(2)
        sentence_ids = torch.arange(N, device=memory.device).repeat_interleave(S)
        sentence_bias = sentence_importance[:, sentence_ids]


        ys = torch.full(
            (B, 1),
            self.vocab.bos_idx,
            dtype=torch.long,
            device=device
        )

        for _ in range(max_len):
            dec_emb = self.pe(self.embed(ys))
            trg_mask = (ys != self.vocab.pad_idx).unsqueeze(1).unsqueeze(2)
            trg_causal_mask = torch.tril(
                torch.ones(ys.size(1), ys.size(1), device=device)
            ).bool()

            out = dec_emb
            for layer in self.decoder_layers:
                out = layer(
                    out,
                    memory,
                    trg_mask,
                    trg_causal_mask,
                    src_mask,
                    sentence_bias
                )


            next_token = self.fc_out(out[:, -1]).argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

            if (next_token == self.vocab.eos_idx).all():
                break

        return ys

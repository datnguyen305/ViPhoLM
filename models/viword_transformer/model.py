import math
import torch
import torch.nn as nn
from vocabs.hierarchy_vocab import Hierarchy_Vocab
from builders.model_builder import META_ARCHITECTURE
from .attention import PhrasalLexemeAttention

class LocalAttention(nn.Module):
    """
    Local (sentence-level) attention sử dụng PhrasalLexemeAttention 
    để trích xuất đặc trưng cụm từ/lexeme trước khi pooling thành vector câu.
    """

    def __init__(self, head, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.head = head
        self.d_kv = hidden_size // head

        # Phrasal Lexeme Attention
        self.phrasal_attn_module = PhrasalLexemeAttention(
            head=head,
            d_model=hidden_size,
            d_q=self.d_kv,
            d_kv=self.d_kv
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        
        # Add value projection to match standard attention mechanism
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attention_mask, phrasal_attn=None):
        """
        x              : (B*N, S, D)
        attention_mask : (B*N, S)  bool (True là valid)
        """
        BN, S, D = x.size()

        # 1. Xử lý mask để tránh NaN cho câu rỗng
        has_any_token = attention_mask.any(dim=-1)
        attention_mask_safe = attention_mask.clone()
        empty_indices = ~has_any_token
        if empty_indices.any():
            attention_mask_safe[empty_indices, 0] = True

        # 2. Sử dụng Phrasal Lexeme Attention để lấy attention weights
        # Module này trả về (attn_matrix, phrasal_attn_scores)
        attn_weights, phrasal_scores = self.phrasal_attn_module(
            x, 
            attention_mask=attention_mask_safe,
            prior_attn=0 if phrasal_attn is None else phrasal_attn
        )

        # 3. Apply attention to values
        # Reshape value for multi-head attention
        value = self.value_proj(x)  # (BN, S, D)
        value = value.view(BN, S, self.head, self.d_kv).permute(0, 2, 1, 3)  # (BN, head, S, d_kv)
        
        # attn_weights is (BN, head, S, S) or (head, S, S), need to handle both cases
        if attn_weights.dim() == 3:
            attn_weights = attn_weights.unsqueeze(0).expand(BN, -1, -1, -1)
        
        # Apply attention
        context = torch.matmul(attn_weights, value)  # (BN, head, S, d_kv)
        context = context.permute(0, 2, 1, 3).contiguous().view(BN, S, D)  # (BN, S, D)
        
        # Output projection
        phrasal_out = self.output_proj(context)

        # Residual connection và Norm
        x = self.norm(x + self.dropout(phrasal_out))

        # 4. Pooling tokens -> sentence representation
        # Sử dụng masked mean pooling để lấy vector đại diện cho câu
        token_mask = attention_mask.unsqueeze(-1).float()  # (BN, S, 1)
        masked_out = x * token_mask 
        
        token_counts = token_mask.sum(dim=1).clamp(min=1.0)
        sent_repr = masked_out.sum(dim=1) / token_counts
        
        # Zero out cho câu rỗng
        sent_repr[empty_indices] = 0.0

        return sent_repr, attn_weights


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
        
        self.sent_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = FeedForward(hidden_size, ffn_hidden, dropout)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # X : (B, N, D)
        # mask : (B, N) with 1 for valid, 0 for padding
        key_padding_mask = ~mask.bool()  # (B, N)

        out, _ = self.sent_attn(
            x, x, x,
            key_padding_mask=key_padding_mask
        )

        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class HierarchicalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.local_attn = LocalAttention(
            head=config.n_head,
            hidden_size=config.hidden_size,
            dropout=config.drop_prob
        )

        self.sent_encoder = nn.ModuleList([
            HierarchicalEncoderLayer(
                hidden_size=config.hidden_size,
                n_head=config.n_head,
                ffn_hidden=config.ffn_hidden,
                dropout=config.drop_prob,
            )
            for _ in range(config.n_layers)
        ])

    def forward(self, x, mask):
        """
        x    : (B, N, S, D)
        mask : (B, N, S) with True/1 for valid, False/0 for padding
        
        Returns:
            sent_repr: (B, N, D)
        """
        B, N, S, D = x.size()

        # ---- token → sentence (Sử dụng Phrasal Lexeme) ----
        x_flat = x.view(B * N, S, D)
        mask_flat = mask.view(B * N, S).bool()

        # Trích xuất đặc trưng câu dựa trên lexeme
        sent_repr, _ = self.local_attn(x_flat, mask_flat)

        sent_repr = sent_repr.view(B, N, D)
        sent_mask = (mask.sum(dim=-1) > 0).float()

        # ---- sentence encoder (Global context giữa các câu) ----
        for layer in self.sent_encoder:
            sent_repr = layer(sent_repr, sent_mask)

        return sent_repr


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, memory_mask):
        """
        query: (B, T, D)
        memory: (B, N, D) - sentence representations
        memory_mask: (B, N) with 1 for valid, 0 for padding
        """
        B, T, D = query.size()
        N = memory.size(1)

        Q = self.q_proj(query)
        K = self.k_proj(memory)
        V = self.v_proj(memory)

        # reshape → multihead
        Q = Q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        K = K.view(B, N, self.n_head, self.d_head).transpose(1, 2)
        V = V.view(B, N, self.n_head, self.d_head).transpose(1, 2)

        # attention score
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores /= math.sqrt(self.d_head)  # (B, H, T, N)

        # mask padding sentences
        if memory_mask is not None:
            scores = scores.masked_fill(
                memory_mask[:, None, None, :] == 0, -1e9
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, dropout):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )

        self.cross_attn = CrossAttention(
            d_model, n_head, dropout
        )

        self.ffn = FeedForward(d_model, ffn_hidden, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, memory_mask):
        """
        x: (B, T, D)
        memory: (B, N, D)
        tgt_mask: (T, T) causal mask
        memory_mask: (B, N)
        """
        # self-attention with causal mask
        _x, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(_x))

        # cross-attention
        _x = self.cross_attn(x, memory, memory_mask)
        x = self.norm2(x + self.dropout(_x))

        # FFN
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

@META_ARCHITECTURE.register()
class ViWordTransformerModel(nn.Module):
    def __init__(self, config, vocab: Hierarchy_Vocab):
        super().__init__()
        self.d_model = config.d_model
        self.embed = nn.Embedding(
            vocab.vocab_size, self.d_model, padding_idx=vocab.pad_idx
        )
        
        self.config = config
        self.device = config.device
        self.MAX_LEN = vocab.max_sentence_length + 2  # +2 for <bos> and <eos>
        self.pe = PositionalEncoding(self.d_model, self.MAX_LEN)

        # use the HierarchicalEncoder wrapper
        self.encoder = HierarchicalEncoder(config.encoder)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                d_model=config.d_model,
                n_head=config.decoder.n_head,
                ffn_hidden=config.decoder.ffn_hidden,
                dropout=config.decoder.drop_prob,
            )
            for _ in range(config.decoder.n_layers)
        ])

        self.fc_out = nn.Linear(self.d_model, vocab.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.vocab = vocab

    def _encode(self, src):
        """Shared encoder logic for forward() and predict().

        src: (B, N, S)  – token ids
        Returns:
            memory      : (B, N, D)  -- sentence representations
            memory_mask : (B, N)
        """
        B, N, S = src.size()

        # Embed and add positional encoding
        src_flat = src.view(B * N, S)
        src_emb = self.embed(src_flat)  # (B*N, S, D)
        src_emb = self.pe(src_emb)  # apply PE
        src_emb = src_emb.view(B, N, S, -1)  # (B, N, S, D)

        src_mask_3d = (src != self.vocab.pad_idx)  # (B, N, S)

        # Call encoder - returns sentence representations
        memory = self.encoder(src_emb, src_mask_3d)

        # Memory mask: which sentences are valid (not all padding)
        memory_mask = (src_mask_3d.sum(dim=-1) > 0).float()  # (B, N)

        return memory, memory_mask

    def forward(self, src, trg):
        """
        src: (B, N_sent, S_token)
        trg: (B, T)         – target including <bos> and <eos>
        """
        trg_input = trg[:, :-1]   # drop last (teacher forcing input)
        trg_label = trg[:, 1:]    # drop first (prediction target)

        memory, memory_mask = self._encode(src)

        # --- decoder ---
        dec_emb = self.pe(self.embed(trg_input))  # (B, T-1, D)

        T = trg_input.size(1)
        # Create causal mask for decoder self-attention
        tgt_causal_mask = torch.triu(
            torch.ones(T, T, device=trg.device, dtype=torch.float32),
            diagonal=1
        ) * -1e9  # Upper triangular with -inf, allowing attention to current and past

        out = dec_emb
        for layer in self.decoder_layers:
            out = layer(out, memory, tgt_causal_mask, memory_mask)

        logits = self.fc_out(out)  # (B, T-1, vocab)
        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            trg_label.reshape(-1),
        )
        
        return logits, loss

    @torch.no_grad()
    def predict(self, src):
        """
        src: (B, N, S) - source token ids
        max_len: maximum generation length (optional)
        """
        self.eval()  # Ensure model is in eval mode
        device = src.device
        B = src.size(0)

        memory, memory_mask = self._encode(src)

        # start with <bos>
        ys = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=device)

        for _ in range(self.MAX_LEN):
            dec_emb = self.pe(self.embed(ys))
            T = ys.size(1)
            tgt_causal_mask = torch.triu(
                torch.ones(T, T, device=device, dtype=torch.float32),
                diagonal=1
            ) * -1e9

            out = dec_emb
            for layer in self.decoder_layers:
                out = layer(out, memory, tgt_causal_mask, memory_mask)

            next_token = self.fc_out(out[:, -1]).argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

            # Stop if all sequences have generated <eos>
            if (next_token == self.vocab.eos_idx).all():
                break

        return ys
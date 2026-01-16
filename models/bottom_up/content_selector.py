import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from builders.model_builder import META_ARCHITECTURE


@META_ARCHITECTURE.register()
class ContentSelector(nn.Module):

    def __init__(self, config, vocab):
        super().__init__()
        self.vocab = vocab
        self.device = torch.device(config.device)

        # ===== Static embedding (vocab riêng) =====
        self.word_embedding = nn.Embedding(
            len(vocab),
            config.word_embed_size,
            padding_idx=vocab.pad_idx
        )

        # ===== PhoBERT =====
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        self.bert = AutoModel.from_pretrained("vinai/phobert-base", add_pooling_layer=False)

        # Register token_type_ids buffer if not present
        if hasattr(self.bert.embeddings, 'token_type_ids'):
            self.bert.embeddings.register_buffer(
                "token_type_ids",
                torch.zeros((1, 512), dtype=torch.long),
                persistent=False
            )

        if config.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # ===== BiLSTM =====
        self.lstm = nn.LSTM(
            input_size=config.word_embed_size + 768,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(config.dropout)

        # Add layer norm for better training stability
        self.layer_norm = nn.LayerNorm(config.hidden_size * 2)

        # Use a small MLP instead of single linear layer
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1)
        )

    def _ids_to_text(self, input_ids):
        """Convert token IDs to text strings."""
        texts = []
        for sent in input_ids:
            toks = [
                self.vocab.itos[i]
                for i in sent.tolist()
                if i not in {
                    self.vocab.pad_idx,
                    self.vocab.bos_idx,
                    self.vocab.eos_idx
                }
            ]
            texts.append(" ".join(toks))
        return texts

    def _phobert_embed(self, input_ids):
        """Generate PhoBERT contextual embeddings with proper handling."""
        texts = self._ids_to_text(input_ids)

        # Tokenize with PhoBERT tokenizer
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
            return_token_type_ids=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Safety check for vocab overflow
        vocab_size = self.bert.config.vocab_size
        if inputs["input_ids"].max() >= vocab_size:
            inputs["input_ids"] = inputs["input_ids"].clamp(0, vocab_size - 1)

        # Ensure token_type_ids match input_ids shape
        if "token_type_ids" not in inputs or inputs["token_type_ids"].shape != inputs["input_ids"].shape:
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

        with torch.no_grad():
            outputs = self.bert(**inputs)

        return outputs.last_hidden_state

    def _align_embeddings(self, static_emb, contextual_emb):
        """Align PhoBERT embeddings with static embeddings by averaging."""
        B, L_static, _ = static_emb.shape
        _, L_context, D_context = contextual_emb.shape

        if L_context == L_static:
            return contextual_emb

        # Create aligned embeddings
        aligned = torch.zeros(B, L_static, D_context, device=static_emb.device)

        for b in range(B):
            if L_context > L_static:
                # More PhoBERT tokens than original (due to subword tokenization)
                ratio = L_context / L_static
                for i in range(L_static):
                    start_idx = int(i * ratio)
                    end_idx = int((i + 1) * ratio)
                    aligned[b, i] = contextual_emb[b, start_idx:end_idx].mean(dim=0)
            else:
                # Fewer PhoBERT tokens than original
                aligned[b, :L_context] = contextual_emb[b]

        return aligned

    def forward(self, input_ids, lengths, return_logits=False):
        """
        Args:
            input_ids: (B, L) tensor of token IDs
            lengths: (B,) tensor of sequence lengths
            return_logits: whether to return raw logits or probabilities
        Returns:
            logits or probabilities for each token being selected
        """
        # Static word embeddings
        static = self.word_embedding(input_ids)  # (B, L, E)

        # PhoBERT contextual embeddings
        contextual = self._phobert_embed(input_ids)  # (B, L_bert, 768)

        # Align contextual embeddings to match static embeddings length
        contextual = self._align_embeddings(static, contextual)  # (B, L, 768)

        # Concatenate both embeddings
        x = torch.cat([static, contextual], dim=-1)  # (B, L, E + 768)
        x = self.dropout(x)

        # Pack sequence for efficient LSTM processing
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)

        # Apply layer norm and dropout
        out = self.layer_norm(out)
        out = self.dropout(out)

        # Compute logits
        logits = self.scorer(out).squeeze(-1)  # (B, L)

        # Mask padding positions
        logits = logits.masked_fill(
            input_ids == self.vocab.pad_idx, -1e9
        )

        return logits if return_logits else torch.sigmoid(logits)


def create_selection_labels(src, tgt, vocab, target_ratio=0.25):
    """
    Create binary labels for content selection.
    Only selects the MOST important tokens, aiming for ~25% selection.
    """
    B, L = src.size()
    labels = torch.zeros(B, L, device=src.device)

    for b in range(B):
        # Remove special tokens
        s = [t for t in src[b].tolist() if t not in {
            vocab.pad_idx, vocab.bos_idx, vocab.eos_idx
        }]
        t = [t for t in tgt[b].tolist() if t not in {
            vocab.pad_idx, vocab.bos_idx, vocab.eos_idx
        }]

        if not s or not t:
            continue

        # Strategy 1: Longest common substring (paper's main approach)
        lcs_idxs = longest_common_substring(s, t)

        # Strategy 2: Token frequency in target (how many times each appears)
        tgt_freq = {}
        for token in t:
            tgt_freq[token] = tgt_freq.get(token, 0) + 1

        # Score each source token based on:
        # 1. Is it in LCS? (high priority)
        # 2. How frequent in target? (secondary priority)
        token_scores = []
        for i, token in enumerate(s):
            score = 0

            # LCS tokens get highest priority
            if i in lcs_idxs:
                score += 10.0

            # Tokens that appear in target get priority by frequency
            if token in tgt_freq:
                score += tgt_freq[token]

            token_scores.append((i, score))

        # Sort by score (descending) and select top tokens
        token_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top tokens up to target_ratio
        target_count = max(1, int(len(s) * target_ratio))

        # Only select tokens with score > 0
        selected_idxs = [
            idx for idx, score in token_scores[:target_count]
            if score > 0
        ]

        for i in selected_idxs:
            if i < L:
                labels[b, i] = 1.0

    return labels


def longest_common_substring(src, tgt):
    """
    Find all positions in src that are part of the longest common substring.
    More conservative - only longest matches.
    """
    m, n = len(src), len(tgt)
    if m == 0 or n == 0:
        return set()

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len, ends = 0, []

    # Build DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src[i - 1] == tgt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    ends = [i]
                elif dp[i][j] == max_len:
                    ends.append(i)

    # Only extract longest common substrings
    idxs = set()

    if max_len >= 2:  # Require at least 2 consecutive tokens
        for e in ends:
            for k in range(max_len):
                idxs.add(e - 1 - k)

    return idxs
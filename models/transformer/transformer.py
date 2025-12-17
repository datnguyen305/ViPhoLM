import torch
from torch import nn
import math
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [B, L, D]
        seq_len = x.size(1)
        return self.pe[:seq_len, :].unsqueeze(0)  # [1, L, D]


@META_ARCHITECTURE.register()
class TransformerModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.src_pad_idx = vocab.pad_idx
        self.trg_pad_idx = vocab.pad_idx
        self.trg_bos_idx = vocab.bos_idx
        self.trg_eos_idx = vocab.eos_idx

        self.d_model = config.d_model
        self.device = config.device
        self.config = config
        self.vocab = vocab

        self.MAX_LENGTH = vocab.max_sentence_length + 2

        # Embedding
        self.src_embedding = nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=self.src_pad_idx)
        self.trg_embedding = nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=self.trg_pad_idx)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_len)
        
        # Dropout for embeddings
        self.dropout = nn.Dropout(config.drop_prob)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.ffn_hidden,
            dropout=config.drop_prob,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.ffn_hidden,
            dropout=config.drop_prob,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)

        # Output projection
        self.output_layer = nn.Linear(config.d_model, vocab.vocab_size)

        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=self.trg_pad_idx)


    # ------------------------- MASKS -------------------------
    def make_src_padding_mask(self, src):
        return (src == self.src_pad_idx)  # [B, L]

    def make_tgt_mask(self, tgt):
        B, T = tgt.size()

        # Padding mask
        padding_mask = (tgt == self.trg_pad_idx)  # [B, T]

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()

        return padding_mask, causal_mask


    # ------------------------- FORWARD -------------------------
    def forward(self, src, tgt):
        config = self.config

        # Trim length
        src = src[:, :config.max_len]
        tgt = tgt[:, :config.max_len]

        # Use tgt[:, :-1] as input, predict tgt[:, 1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Masks for input sequence
        src_key_padding = self.make_src_padding_mask(src)
        tgt_padding, tgt_causal = self.make_tgt_mask(tgt_input)

        # Embedding + Positional encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.trg_embedding(tgt_input) * math.sqrt(self.d_model)

        
        src_pos = self.pos_encoding(src_emb)
        tgt_pos = self.pos_encoding(tgt_emb)

        
        enc_input = self.dropout(src_emb + src_pos)
        dec_input = self.dropout(tgt_emb + tgt_pos)

        # Encoder
        memory = self.encoder(
            enc_input,
            src_key_padding_mask=src_key_padding
        )  # [B, L, D]

        # Decoder
        out = self.decoder(
            dec_input,
            memory,
            tgt_mask=tgt_causal,
            tgt_key_padding_mask=tgt_padding,
            memory_key_padding_mask=src_key_padding
        )

        logits = self.output_layer(out)  # [B, T-1, V]

        # Compute loss on shifted target
        loss = self.loss(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        return logits, loss

    def predict(self, src: torch.Tensor):
        self.eval()
        
        config = self.config
    
        src = src[:, :config.max_len]
        src_key_padding = self.make_src_padding_mask(src)
    
        # Embedding + Positional encoding
        B = src.size(0)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_pos = self.pos_encoding(src_emb)
        enc_input = src_emb + src_pos
        
        # Use no_grad for inference
        with torch.no_grad():
            memory = self.encoder(enc_input, src_key_padding_mask=src_key_padding)
        
            # Start with BOS
            tgt_seq = torch.full((B, 1), self.trg_bos_idx, device=src.device, dtype=torch.long)
            finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        
            for _ in range(self.MAX_LENGTH):
                tgt_padding, tgt_causal = self.make_tgt_mask(tgt_seq)
        
                # Embedding + Positional encoding
                tgt_emb = self.trg_embedding(tgt_seq) * math.sqrt(self.d_model)
                tgt_pos = self.pos_encoding(tgt_emb)
                dec_input = tgt_emb + tgt_pos
        
                dec_out = self.decoder(
                    dec_input,
                    memory,
                    tgt_mask=tgt_causal,
                    tgt_key_padding_mask=tgt_padding,
                    memory_key_padding_mask=src_key_padding
                )
        
                # Get logits for last position
                logits = self.output_layer(dec_out[:, -1, :])  # [B, vocab]
                next_token = logits.argmax(dim=-1, keepdim=True)
        
                # Append next token
                tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
        
                # Check if finished
                finished |= (next_token.squeeze(1) == self.trg_eos_idx)
        
                if finished.all():
                    break
        
        return tgt_seq[:, 1:]  # Remove BOS
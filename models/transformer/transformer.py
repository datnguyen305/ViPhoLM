import torch
from torch import nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


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

        # Embedding
        self.src_embedding = nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=self.src_pad_idx)
        self.trg_embedding = nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=self.trg_pad_idx)

        # Positional encoding
        self.pos_encoder = nn.Embedding(config.max_len, config.d_model)
        self.pos_decoder = nn.Embedding(config.max_len, config.d_model)

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

        self.loss = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)


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

        # FIXED: Use tgt[:-1] as input, predict tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Masks for input sequence
        src_key_padding = self.make_src_padding_mask(src)
        tgt_padding, tgt_causal = self.make_tgt_mask(tgt_input)

        # Positional encoding - create for actual sequence length
        B = src.size(0)
        src_len = src.size(1)
        tgt_len = tgt_input.size(1)
        
        src_pos = self.pos_encoder(torch.arange(src_len, device=src.device)).unsqueeze(0)
        tgt_pos = self.pos_decoder(torch.arange(tgt_len, device=tgt_input.device)).unsqueeze(0)

        enc_input = self.src_embedding(src) + src_pos
        dec_input = self.trg_embedding(tgt_input) + tgt_pos

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
    
        # Positional encoding for source
        B = src.size(0)
        src_len = src.size(1)
        
        # No need to expand, broadcasting handles it
        src_pos = self.pos_encoder(torch.arange(src_len, device=src.device)).unsqueeze(0)
    
        enc_input = self.src_embedding(src) + src_pos
        
        # use no_grad for inference
        with torch.no_grad():
            memory = self.encoder(enc_input, src_key_padding_mask=src_key_padding)
        
            # Start with BOS
            tgt_seq = torch.full((B, 1), self.trg_bos_idx, device=src.device, dtype=torch.long)
            finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        
            for _ in range(config.max_len):
                tgt_len = tgt_seq.size(1)
                tgt_padding, tgt_causal = self.make_tgt_mask(tgt_seq)
        
                # Create positional encoding for current length
                tgt_pos = self.pos_decoder(torch.arange(tgt_len, device=src.device)).unsqueeze(0)
        
                dec_input = self.trg_embedding(tgt_seq) + tgt_pos
        
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
        
        return tgt_seq[:, 1:]   # remove BOS
import torch
from torch import nn

import math

from builders.model_builder import META_ARCHITECTURE
from models.transformer.utils import PositionalEncoding
from vocabs.subword import Dual_Subword_Vocab
from vocabs.viword_vocab import ViWordVocab

@META_ARCHITECTURE.register()
class PhonemeTransformer(nn.Module):
    def __init__(self, config, subword_vocab: Dual_Subword_Vocab, phoneme_vocab: ViWordVocab):
        super().__init__()

        self.subword_vocab = subword_vocab
        self.phoneme_vocab = phoneme_vocab

        self.src_pad_idx = subword_vocab.pad_idx
        self.trg_pad_idx = subword_vocab.pad_idx
        
        self.trg_bos_idx = phoneme_vocab.bos_idx
        self.trg_eos_idx = phoneme_vocab.eos_idx

        self.d_model = config.d_model
        self.device = config.device
        self.config = config

        self.src_max_len = subword_vocab.max_sentence_length[config.src_lang] + 2
        self.tgt_max_len = phoneme_vocab.max_sentence_length + 2

        self.src_vocab_size = subword_vocab.src_vocab_size
        self.tgt_vocab_size = phoneme_vocab.vocab_size

        # Embedding
        self.src_embedding = nn.Embedding(self.src_vocab_size, config.d_model, padding_idx=self.src_pad_idx)
        self.trg_embedding = nn.Embedding(self.tgt_vocab_size, config.d_model, padding_idx=self.trg_pad_idx)
        self.trg_emb_fc = nn.Linear(
            in_features=config.d_model*3,
            out_features=config.d_model
        )

        # Positional encoding
        self.src_pos_encoding = PositionalEncoding(config.d_model, subword_vocab.max_sentence_length[config.src_lang] + 2)
        self.tgt_pos_encoding = PositionalEncoding(config.d_model, phoneme_vocab.max_sentence_length + 2)
        
        # Dropout for embeddings
        self.dropout = nn.Dropout(config.dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.head,
            dim_feedforward=config.dff,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder.layer_dim)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.head,
            dim_feedforward=config.dff,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder.layer_dim)

        # Output projection
        self.initial_fc = nn.Linear(
            in_features=config.d_model,
            out_features=phoneme_vocab.vocab_size
        )
        self.rhyme_fc = nn.Linear(
            in_features=config.d_model,
            out_features=phoneme_vocab.vocab_size
        )
        self.tone_fc = nn.Linear(
            in_features=config.d_model,
            out_features=phoneme_vocab.vocab_size
        )

        self.loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing, ignore_index=self.trg_pad_idx)

    def make_src_padding_mask(self, src: torch.Tensor):
        if src.dim() == 3:
            src = src[:, :, 0]

        return (src == self.src_pad_idx)  # [B, L]

    def make_tgt_mask(self, tgt: torch.Tensor):
        if tgt.dim() == 3:
            tgt = tgt[:, :, 0]

        _, T = tgt.size()

        # Padding mask
        padding_mask = (tgt == self.trg_pad_idx)  # [B, T]

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()

        return padding_mask, causal_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # Use tgt[:-1] as input, predict tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Masks for input sequence
        src_key_padding = self.make_src_padding_mask(src)
        tgt_padding, tgt_causal = self.make_tgt_mask(tgt_input)

        # Embedding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model) # (bs, len, 3, dim)

        tgt_emb = self.trg_embedding(tgt_input) # (bs, len, 3, dim)
        bs, l, _, _ = tgt_emb.shape
        tgt_emb = tgt_emb.reshape(bs, l, -1) # (bs, len, 3*dim)
        tgt_emb = self.trg_emb_fc(tgt_emb) * math.sqrt(self.d_model) # (bs, len, dim)

        # Positional Embedding
        src_pos = self.src_pos_encoding(src_emb)
        tgt_pos = self.tgt_pos_encoding(tgt_emb)
        
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

        initial_logits = self.initial_fc(out)
        rhyme_logits = self.rhyme_fc(out)
        tone_logits = self.tone_fc(out)
        
        # get the components
        initial_tgt_outputs = tgt_output[:, :, 0]
        rhyme_tgt_outputs = tgt_output[:, :, 1]
        tone_tgt_outputs = tgt_output[:, :, 2]

        # Compute loss on shifted target
        initial_loss = self.loss(initial_logits.reshape(-1, self.phoneme_vocab.vocab_size), initial_tgt_outputs.reshape(-1))
        rhyme_loss = self.loss(rhyme_logits.reshape(-1, self.phoneme_vocab.vocab_size), rhyme_tgt_outputs.reshape(-1))
        tone_loss = self.loss(tone_logits.reshape(-1, self.phoneme_vocab.vocab_size), tone_tgt_outputs.reshape(-1))
        loss = initial_loss + rhyme_loss + tone_loss

        return (initial_logits, rhyme_logits, tone_logits), loss

    def predict(self, src: torch.Tensor):
        self.eval()
    
        # Masks for input sequence
        src_key_padding = self.make_src_padding_mask(src)

        # Embedding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model) # (bs, len, 3, dim)
        bs, l, _ = src_emb.shape

        # Positional Embedding
        src_pos = self.src_pos_encoding(src_emb)
        
        enc_input = self.dropout(src_emb + src_pos)
        
        # Use no_grad for inference
        with torch.no_grad():
            memory = self.encoder(enc_input, src_key_padding_mask=src_key_padding)
        
            # Start with BOS
            tgt_seq = torch.full((bs, 3), self.trg_bos_idx, device=src.device, dtype=torch.long)
            finished = torch.zeros(bs, dtype=torch.bool, device=src.device)
        
            for _ in range(self.tgt_max_len):
                tgt_padding, tgt_causal = self.make_tgt_mask(tgt_seq)
        
                # Embedding + Positional encoding
                tgt_emb = self.trg_embedding(src) # (bs, len, 3, dim)
                bs, l, _, _ = tgt_emb.shape
                tgt_emb = tgt_emb.reshape(bs, l, -1) # (bs, len, 3*dim)
                tgt_emb = self.trg_emb_fc(src_emb)* math.sqrt(self.d_model) # (bs, len, dim)

                tgt_pos = self.tgt_pos_encoding(tgt_emb)
                dec_input = self.dropout(tgt_emb + tgt_pos)
        
                dec_out = self.decoder(
                    dec_input,
                    memory,
                    tgt_mask=tgt_causal,
                    tgt_key_padding_mask=tgt_padding,
                    memory_key_padding_mask=src_key_padding
                )
        
                # Get logits for last position
                initial_logits = self.initial_fc(dec_out)
                rhyme_logits = self.rhyme_fc(dec_out)
                tone_logits = self.tone_fc(dec_out)
                
                next_initial = initial_logits.argmax(dim=-1, keepdim=True)
                next_rhyme = rhyme_logits.argmax(dim=-1, keepdim=True)
                next_tone = tone_logits.argmax(dim=-1, keepdim=True)
                next_token = torch.stack([next_initial, next_rhyme, next_tone], dim=-1).unsqueeze(1) # (bs, 1, 3)
        
                # Append next token
                tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
        
                # Check if finished
                finished |= (next_token.mean().int() == self.trg_eos_idx)
        
                if finished.all():
                    break
        
        return tgt_seq[:, 1:]  # Remove BOS
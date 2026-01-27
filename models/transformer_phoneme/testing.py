import torch
from torch import nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE
from .modules.pe import PositionalEncoding
from .blocks.encoder_block import TransformerEncoderBlock
from .blocks.decoder_block import TransformerDecoderBlock
from .utils.casual_mask import create_causal_mask
from .utils.pad_mask import create_padding_mask
@META_ARCHITECTURE.register()
class Testing(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx=vocab.pad_idx)
        self.output_embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx=vocab.pad_idx)
        self.encoder = TransformerEncoderBlock(config.encoder, vocab)
        self.decoder = TransformerDecoderBlock(config.decoder, vocab)
        self.PE = PositionalEncoding(config.hidden_size, max_len=5000)
        self.config = config
        
        self.MAX_LENGTH = vocab.max_sentence_length + 2
        self.vocab = vocab
        self.d_model = config.d_model

        self.max_length = config.max_length
        self.loss = nn.CrossEntropyLoss()
        self.fc_out = nn.Linear(config.hidden_size, vocab.vocab_size)

    def forward(self, src, trg):
        config = self.config
        device = src.device

        # 1. Xác định độ dài mục tiêu chung (S)
        # Thường là giá trị nhỏ hơn giữa max thực tế và config.max_len
        max_seq_len = max(src.size(1), trg.size(1) - 1)
        S = min(config.max_length, max_seq_len)

        def pad_to_length(t, length):
            curr = t.size(1)
            if curr < length:
                padding = torch.full((t.size(0), length - curr), self.vocab.pad_idx, dtype=torch.long, device=device)
                return torch.cat([t, padding], dim=1)
            return t[:, :length]

        src = pad_to_length(src, S)
        trg = pad_to_length(trg, S + 1)

        # Cắt chuỗi cho training
        trg_input = trg[:, :-1]
        trg_label = trg[:, 1:]

        # Embedding + Positional Encoding
        src_emb = self.PE(self.input_embedding(src))
        trg_emb = self.PE(self.output_embedding(trg_input))

        # Masking
        src_mask = create_padding_mask(src, self.vocab.pad_idx).to(src.device)
        trg_mask = create_padding_mask(trg_input, self.vocab.pad_idx).to(trg_input.device)
        trg_causal_mask = create_causal_mask(trg_input.size(1), device=trg.device)

        # Encoder - Decoder
        encoder_outs = self.encoder(src_emb, src_mask=src_mask)
        outs = self.decoder(trg_emb, encoder_outs, trg_mask=trg_mask, 
                            trg_causal_mask=trg_causal_mask, src_mask=src_mask)
        
        logits = self.fc_out(outs)
        loss = self.loss(logits.reshape(-1, self.vocab.vocab_size), trg_label.reshape(-1))
        
        return logits, loss
    
    def predict(self, src):
        config = self.config
        device = src.device
        B = src.size(0)
        
        # --- BƯỚC 1: ĐỒNG BỘ KÍCH THƯỚC (QUAN TRỌNG) ---
        # Để Cross-Attention chạy được Phrasal_Lexeme, Src và Trg phải cùng độ dài.
        # Ta lấy max_length làm chuẩn chung.
        target_len = self.config.max_length

        # Xử lý SRC: Ép về đúng độ dài target_len
        if src.size(1) < target_len:
            pad_len = target_len - src.size(1)
            src = torch.cat([src, torch.full((B, pad_len), self.vocab.pad_idx, device=device)], dim=1)
        else:
            src = src[:, :target_len]

        # --- BƯỚC 2: CHẠY ENCODER ---
        src_emb = self.PE(self.input_embedding(src))
        src_mask = create_padding_mask(src, self.vocab.pad_idx).to(device)
        encoder_outs = self.encoder(src_emb, src_mask=src_mask)

        # --- BƯỚC 3: CHUẨN BỊ DECODER (FIXED BUFFER) ---
        # Tạo sẵn bộ đệm cố định, không dùng append/cat
        decoder_input = torch.full((B, target_len), self.vocab.pad_idx, dtype=torch.long, device=device)
        decoder_input[:, 0] = self.vocab.bos_idx 

        # --- BƯỚC 4: VÒNG LẶP SINH TỪ ---
        # Chỉ chạy đến target_len - 1
        for i in range(target_len - 1):
            # Mask luôn cố định kích thước
            trg_mask = create_padding_mask(decoder_input, self.vocab.pad_idx).to(device)
            trg_causal_mask = create_causal_mask(target_len, device=device)

            # Forward toàn bộ chuỗi (kích thước luôn là target_len)
            trg_emb = self.PE(self.output_embedding(decoder_input))
            outs = self.decoder(trg_emb, encoder_outs, 
                                trg_mask=trg_mask, 
                                trg_causal_mask=trg_causal_mask, 
                                src_mask=src_mask)
            
            # SỬA LỖI LOGIC: Lấy output tại đúng vị trí i, không phải -1
            # Tại bước i, ta dùng thông tin từ 0..i để dự đoán i+1
            current_out = outs[:, i:i+1, :] 
            logits = self.fc_out(current_out) 
            next_token = logits.argmax(dim=-1) # (B, 1)

            # SỬA LỖI CRASH: Gán trực tiếp vào buffer, KHÔNG dùng torch.cat
            decoder_input[:, i+1] = next_token.squeeze(-1)

            # Kiểm tra dừng sớm (chỉ hiệu quả với batch=1)
            if B == 1 and next_token.item() == self.vocab.eos_idx:
                return decoder_input[:, :i+2]

        return decoder_input
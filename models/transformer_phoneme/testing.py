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
        
        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2 # +2 for BOS and EOS tokens
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
        # Hàm predict được viết lại để luôn duy trì Sq == Sk
        device = src.device
        batch_size = src.size(0)
        pad_idx = self.vocab.pad_idx
        
        # 1. Chuẩn bị Encoder
        # Pad src về max_length hoặc giữ nguyên, nhưng decoder phải theo độ dài này
        S = src.size(1) 
        
        src_emb = self.PE(self.input_embedding(src))
        src_mask = create_padding_mask(src, pad_idx).to(device)
        encoder_outs = self.encoder(src_emb, src_mask=src_mask)

        # 2. Chuẩn bị Decoder Input CỐ ĐỊNH KÍCH THƯỚC (QUAN TRỌNG)
        # Thay vì list append, ta tạo tensor full kích thước S ngay từ đầu
        # Để đảm bảo Cross-Attention luôn là S x S
        decoder_input = torch.full((batch_size, S), pad_idx, dtype=torch.long, device=device)
        decoder_input[:, 0] = self.vocab.bos_idx # Gán token đầu tiên

        # 3. Vòng lặp sinh từ
        # Chỉ chạy đến S-1 vì token cuối cùng không dùng để dự đoán tiếp
        for i in range(S - 1):
            # Tạo mask
            trg_mask = create_padding_mask(decoder_input, pad_idx)
            trg_causal_mask = create_causal_mask(S, device=device)

            # Forward Decoder (Luôn đưa cả chuỗi dài S vào)
            trg_emb = self.PE(self.output_embedding(decoder_input))
            
            outs = self.decoder(trg_emb, encoder_outs, 
                                trg_mask=trg_mask, 
                                trg_causal_mask=trg_causal_mask, 
                                src_mask=src_mask)
            
            # Lấy logit tại bước thứ i để dự đoán bước i+1
            # logits shape: (Batch, 1, Vocab)
            current_logits = self.fc_out(outs[:, i:i+1, :]) 
            next_token = current_logits.argmax(dim=-1) # (Batch, 1)

            # Cập nhật vào decoder_input tại vị trí i+1
            decoder_input[:, i+1] = next_token.squeeze(-1)

            # (Tùy chọn) Kiểm tra dừng sớm nếu batch=1
            if batch_size == 1 and next_token.item() == self.vocab.eos_idx:
                break
        
        return decoder_input
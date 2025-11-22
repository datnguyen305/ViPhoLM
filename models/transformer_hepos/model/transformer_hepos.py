import torch
import torch.nn as nn

from models.transformer_hepos.model.encoder import Encoder
from models.transformer_hepos.model.decoder import Decoder
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class TransformerHeposModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.src_pad_idx = vocab.pad_idx
        self.trg_pad_idx = vocab.pad_idx
        self.trg_bos_idx = vocab.bos_idx
        self.trg_eos_idx = vocab.eos_idx
        self.vocab_size = vocab.vocab_size
        self.device = config.device
        self.config = config
        self.d_model = config.d_model  # Thêm attribute này

        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)

    def forward(self, src, trg):
        config = self.config
        if src.shape[1] > config.max_len:
            src = src[:, :config.max_len]
        if trg.shape[1] > config.max_len:
            trg = trg[:, :config.max_len]

        # trg is shifted_right_label (target for loss)
        # Tạo decoder input bằng cách thêm BOS vào đầu và bỏ token cuối
        batch_size = trg.size(0)
        bos_tokens = torch.full((batch_size, 1), self.trg_bos_idx, dtype=torch.long, device=trg.device)
        decoder_input = torch.cat([bos_tokens, trg[:, :-1]], dim=1)  # [BOS, token1, ..., tokenN]
        
        # Đảm bảo decoder_input và trg có cùng độ dài
        max_len = min(decoder_input.size(1), trg.size(1))
        decoder_input = decoder_input[:, :max_len]
        trg = trg[:, :max_len]

        # Validation target
        invalid_mask = (trg < 0) | (trg >= self.vocab_size)
        if invalid_mask.any():
            print("Có target không hợp lệ! Thay thế bằng pad_idx.")
        trg = torch.where(invalid_mask, self.trg_pad_idx, trg)

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(decoder_input)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(decoder_input, enc_src, trg_mask, src_mask)

         # Tính loss với shifted target
        output_flat = output.contiguous().view(-1, output.size(-1))  # [B*T, Vocab]
        trg_flat = trg.contiguous().view(-1)                         # [B*T]

        loss = self.loss_fn(output_flat, trg_flat)

        return output, loss

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, device=trg.device)).bool()
        return trg_pad_mask & trg_sub_mask

    def predict(self, src: torch.Tensor) -> torch.Tensor:
        config = self.config

        # Cắt src nếu quá dài
        if src.shape[1] > config.max_len:
            src = src[:, :config.max_len]

        # Tạo mask cho chuỗi nguồn
        src_mask = self.make_src_mask(src)
        
        # Lấy đầu ra từ bộ mã hóa
        enc_src = self.encoder(src, src_mask)
        
        # Chuẩn bị đầu vào ban đầu cho bộ giải mã là token BOS
        batch_size = src.size(0)
        decoder_input = torch.full((batch_size, 1), self.trg_bos_idx, dtype=torch.long, device=src.device)
        
        # Khởi tạo trạng thái ẩn cho bộ giải mã
        decoder_hidden = None
        outputs = []

        # Tạo dự đoán từng bước
        for _ in range(config.max_len):
            # Tạo mask cho chuỗi đích
            trg_mask = self.make_trg_mask(decoder_input)
            
            # Lấy đầu ra của bộ giải mã
            decoder_output = self.decoder(decoder_input, enc_src, trg_mask, src_mask)
            
            # Lấy token có xác suất cao nhất (argmax) từ đầu ra của bộ giải mã
            # Chỉ lấy token cuối cùng (vị trí -1)
            next_token = decoder_output[:, -1:, :].argmax(dim=-1)  # (batch_size, 1)
            
            # Thêm token dự đoán vào chuỗi kết quả
            outputs.append(next_token)
            
            # Cập nhật decoder_input cho bước tiếp theo
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Nếu gặp token EOS, dừng lại
            if (next_token == self.trg_eos_idx).all():
                break

        # Nối tất cả các token dự đoán thành một tensor duy nhất
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
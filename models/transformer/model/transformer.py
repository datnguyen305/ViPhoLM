import torch
from torch import nn

from models.transformer.model.decoder import Decoder
from models.transformer.model.encoder import Encoder

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
        
        self.encoder = Encoder(config.encoder, vocab)

        self.decoder = Decoder(config.decoder, vocab)

        self.d_model = config.d_model
        self.device = config.device 
        self.config = config
        self.vocab = vocab

        self.loss = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)


    def forward(self, src, trg):
        config = self.config
        
        # Cắt src nếu quá dài
        if src.shape[1] > config.max_len:
            src = src[:, :config.max_len]

        # Cắt trg nếu quá dài
        if trg.shape[1] > config.max_len:
            trg = trg[:, :config.max_len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)

         # Tính loss
        output_flat = output.contiguous().view(-1, output.size(-1))  # [B*T, Vocab]
        trg_flat = trg.contiguous().view(-1)                         # [B*T]

        loss = self.loss(output_flat, trg_flat)


        return output, loss

    def make_src_mask(self, src):
        # max_seq_len = src.shape[1]  # Lấy độ dài thực tế
        # src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = src_mask[:, :, :max_seq_len, :max_seq_len]  # Cắt mask phù hợp
        # return src_mask

        """src: [B, src_len]
        return: [B,1,1,src_len] bool mask (True = not pad)"""
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask


    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        # trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask # [B,1,T,T]
        return trg_mask
    
    def predict(self, src: torch.Tensor) -> torch.Tensor:
        config = self.config

        # Cắt src nếu quá dài
        if src.shape[1] > config.max_len:
            src = src[:, :config.max_len]

        # Tạo mask cho chuỗi nguồn
        src_mask = self.make_src_mask(src)
        
        # Lấy đầu ra từ bộ mã hóa
        enc_src = self.encoder(src, src_mask)   # [B, src_len, d_model]
        
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
            
            # Lấy từ mới nhất
            next_token = decoder_output[:, -1, :].argmax(dim=-1, keepdim=True)
            print("Step token:", next_token.squeeze().tolist())

            # Append
            outputs.append(next_token)

            # Dùng nó làm input cho bước tiếp theo
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Nếu gặp token EOS, dừng lại
            if (next_token == self.trg_eos_idx).all():
                break

        # Nối tất cả các token dự đoán thành một tensor duy nhất
        outputs = torch.cat(outputs, dim=1)
        
        return outputs

import torch
from torch import nn
from vocabs.viword_vocab import Vocab
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE
from models.longformer.utils.padding_mask import create_padding_mask, create_standard_padding_mask
from models.longformer.utils.causal_mask import create_causal_mask
from models.longformer.blocks.decoder_block import TransformerDecoderBlock
from models.longformer.blocks.encoder_block import TransformerEncoderBlock
from models.longformer.layers.phoneme_feed_forward import FeedForward
from models.longformer.embedding.positional_embedding import PositionalEncoding

@META_ARCHITECTURE.register()
class Longformer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.d_model = config.d_model
        self.MAX_LENGTH = config.inference_length
        self.config = config

        # Positional Encoding
        self.PE = PositionalEncoding(self.d_model, max_len=self.config.max_len + 10)

        # --- Encoder ---
        self.src_embedding = nn.Embedding(vocab.vocab_size, config.d_model)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoderBlock(config, self.vocab)

        # --- Decoder ---
        self.tgt_embedding = nn.Embedding(vocab.vocab_size, config.d_model)
        self.decoder = TransformerDecoderBlock(config, self.vocab)
        
        # Output layers
        self.outs = nn.Linear(config.d_model, vocab.vocab_size)
        
        # Loss function (Nên ignore padding thay vì unk_idx để model không học cách predict padding)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def forward(self, src, trg):
        # src: (B, S)
        # trg: (B, S_trg)
        B, S = src.shape
        src = src[:, :self.config.max_len]
        trg = trg[:, :self.config.max_len]

        # Padding src lên config.max_len bằng vocab.pad_idx
        if src.shape[1] < self.config.max_len:
            pad_length = self.config.max_len - src.shape[1]
            pad = torch.full((B, pad_length), self.vocab.pad_idx, device=src.device, dtype=torch.long)
            src = torch.cat([src, pad], dim=1)

        # Padding trg lên config.max_len bằng vocab.pad_idx
        if trg.shape[1] < self.config.max_len:
            pad_length = self.config.max_len - trg.shape[1]
            pad = torch.full((B, pad_length), self.vocab.pad_idx, device=trg.device, dtype=torch.long)
            trg = torch.cat([trg, pad], dim=1) # Đã sửa lỗi ghi đè vào src

        # Lưu ý: Cần đảm bảo hàm create_padding_mask của bạn nhận tham số pad_idx thay vì số 3 fix cứng
        encoder_padding_mask = create_padding_mask(src, self.vocab.pad_idx)
        
        # Shift target và decoder_input
        target = trg[:, 1:]        # (B, S - 1)
        decoder_input = trg[:, :-1] # (B, S - 1)
    
        # --- Chạy Encoder ---
        x_src = self.dropout(self.src_embedding(src)) # (B, S, d_model)
        x_src = self.PE(x_src)
        memory = self.encoder(x_src, encoder_padding_mask) # (B, S, d_model)

        # --- Chạy Decoder ---
        decoder_padding_mask = create_standard_padding_mask(decoder_input, self.vocab.pad_idx)
        decoder_causal_mask = create_causal_mask(decoder_input.size(1), self.config.device)
        memory_padding_mask_bool = create_standard_padding_mask(src, self.vocab.pad_idx)

        x_tgt = self.dropout(self.tgt_embedding(decoder_input)) # (B, S-1, d_model)
        x_tgt = self.PE(x_tgt)

        dec_out = self.decoder(x_tgt, memory, decoder_causal_mask, \
                               decoder_padding_mask, memory_padding_mask_bool)
        
        # Output Projection
        logits = self.outs(dec_out)        # (B, S-1, vocab_size)

        # Tính Loss trực tiếp
        # logits.view: (B * (S-1), vocab_size) | target.reshape: (B * (S-1))
        total_loss = self.loss_fn(logits.view(-1, self.vocab.vocab_size), target.reshape(-1))

        return 0, total_loss 
    
    def predict(self, src):
        # src: (B, S)
        src = src[:, :self.config.max_len]
        B, S = src.shape 

        # Padding src
        if src.shape[1] < self.config.max_len:
            pad_length = self.config.max_len - src.shape[1]
            pad = torch.full((B, pad_length), self.vocab.pad_idx, device=src.device, dtype=torch.long)
            src = torch.cat([src, pad], dim=1)

        encoder_padding_mask = create_padding_mask(src, self.vocab.pad_idx)
        memory_padding_mask_bool = create_standard_padding_mask(src, self.vocab.pad_idx)
        
        # --- Chạy Encoder ---
        x_src = self.dropout(self.src_embedding(src))
        x_src = self.PE(x_src)
        memory = self.encoder(x_src, encoder_padding_mask)

        # --- Khởi tạo Decoder Input ---
        # Bắt đầu bằng [<BOS>]
        decoder_input = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=self.config.device)

        outputs = []
        
        # --- Vòng lặp sinh Text ---
        for _ in range(self.MAX_LENGTH): 
            x_tgt = self.dropout(self.tgt_embedding(decoder_input))
            x_tgt = self.PE(x_tgt)

            # Masking
            trg_mask = create_standard_padding_mask(decoder_input, self.vocab.pad_idx)
            trg_causal_mask = create_causal_mask(decoder_input.size(1), self.config.device)

            dec_out = self.decoder(x_tgt, memory, trg_causal_mask, \
                                   trg_mask, memory_padding_mask_bool)
            
            # Chỉ lấy dự đoán của token cuối cùng
            last_token_out = dec_out[:, -1:, :] # (B, 1, d_model)
            
            logits = self.outs(last_token_out) # (B, 1, vocab_size)
            
            next_token = logits.argmax(dim=-1) # (B, 1)
            outputs.append(next_token)
            
            # Cập nhật decoder_input cho bước tiếp theo
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # Dừng sớm nếu batch size = 1 và sinh ra <EOS>
            if B == 1 and next_token.item() == self.vocab.eos_idx:
                break
                
        outputs = torch.cat(outputs, dim=1) # (B, MAX_LENGTH_SINH_RA)

        return outputs
import torch
from torch import nn
from vocabs.viword_vocab import Vocab
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE
from models.vipholm.utils.clone import clones
from models.vipholm.utils.padding_mask import create_padding_mask, create_standard_padding_mask
from models.vipholm.utils.causal_mask import create_causal_mask
from models.vipholm.blocks.decoder_block import TransformerDecoderBlock
from models.vipholm.blocks.encoder_block import TransformerEncoderBlock
from models.vipholm.layers.phoneme_feed_forward import FeedForward
from models.vipholm.embedding.positional_embedding import PositionalEncoding

@META_ARCHITECTURE.register()
class ViPhoLM(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.d_model = config.d_model
        self.MAX_LENGTH = self.vocab.max_sentence_length + 2 
        self.config = config

        # Positional Encoding
        self.PE = PositionalEncoding(self.d_model, max_len=self.config.max_len + 10)

        # Encoder 
        self.src_embedding = nn.Embedding(vocab.vocab_size, config.d_model)
        self.linear = nn.Linear(config.d_model * 3, config.d_model)
        self.dropout = nn.Dropout(0.1)

        self.encoder = TransformerEncoderBlock(config, self.vocab)

        # Decoder  
        self.tgt_embedding = nn.Embedding(vocab.vocab_size, config.d_model)
        self.decoder = TransformerDecoderBlock(config, self.vocab)
        self.out = nn.Linear(config.d_model, vocab.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.unk_idx)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
                
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
            
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, src, trg):
        # src: (B, S, 3)
        # trg: (B, S)
        src = src[:, :self.config.max_len]
        trg = trg[:, :self.config.max_len]

        # Padding to config.max_len 
        # src (B, S, 3) with S < config.max_len
        if src.shape[1] < self.config.max_len:
            pad_length = self.config.max_len - src.shape[1]

            pad = torch.zeros(src.shape[0], pad_length, 3,\
                               device=src.device, dtype=torch.long)
            pad[:,:,0] = 3

            src = torch.cat([src, pad], dim=1)

        # trg (B, S) with S < config.max_len
        if trg.shape[1] < self.config.max_len:
            pad_length = self.config.max_len - trg.shape[1]

            pad = torch.zeros(trg.shape[0], pad_length,\
                               device=trg.device, dtype=torch.long)
            
            trg = torch.cat([trg, pad], dim=1)

        encoder_padding_mask = create_padding_mask(src)
        

        target = trg[:, 1:]
        # target: (B, S - 1) [4, 5, 8, ... <eos>]
        
        decoder_input = trg[:, :-1]
        # decoder_input: (B, S - 1) [<bos>, 3, 4, 5, 6, ...]
    
        # src: (B, S, 3)
        embeds = self.dropout(self.src_embedding(src))
        # embeds: (B, S, 3, d_model)
        B, S, _ = src.shape
        
        input = embeds.reshape(B, S, -1) 
        # input: (B, S, 3*d_model)
        
        input = self.linear(input)
        # input: (B, S, 3*d_model) -> (B, S, d_model)
        
        # Positional Encoding
        input = self.PE(input)
        # input: (B, S, d_model)
        
        memory = self.encoder(
            input, 
            encoder_padding_mask
        )
        # memory: (B, S, d_model)
        
        
        #  DECODER
        
        # Decoder padding
        B, S = decoder_input.shape
        decoder_padding_mask = create_standard_padding_mask(decoder_input, 3)
        decoder_causal_mask = create_causal_mask(S, self.config.device)
        memory_padding_mask_bool = create_standard_padding_mask(src, 3)
        
        """
            # trg: (B, S), Ex: combinations("t","ɯŋ","˧˩") -> tửng
            # decoder_input: (B, S - 1, 3), Ex: [<bos>, 3, 4, 5, 6, ...]
            # target: (B, S - 1, 3) [4, 5, 8, ... <eos>]
        """
        
        embeds = self.dropout(self.tgt_embedding(decoder_input))
        # embeds: (B, S, d_model)
        
        # Positional Encoding
        input = self.PE(embeds)
        # input: (B, S, d_model)

        logits = self.decoder(input, memory, decoder_causal_mask, \
                         decoder_padding_mask, memory_padding_mask_bool)
        # logits: (B, S, d_model)
        
        out = self.out(logits)
        # out: (B, S -1, vocab_size)
        # target: (B, S - 1)
        
        loss = self.loss(out.reshape(-1, self.vocab.vocab_size), target.reshape(-1))
        

        return 0, loss
    def predict(self, src, max_len=None):
        """
        Hàm inference (dự đoán) sinh ra câu đích từ câu nguồn.
        src: (B, S, 3) - Tensor chứa âm tiết đầu vào đã tách thành (Initial, Rhyme, Tone).
        """
        # Chuyển mô hình sang chế độ đánh giá (tắt Dropout)
        self.eval()
        
        device = src.device
        B, S, _ = src.shape
        if max_len is None:
            max_len = self.config.max_len

        with torch.no_grad():
            # ==========================================
            # 1. ENCODER PASS (Chỉ chạy 1 lần)
            # ==========================================
            # Cắt src nếu dài hơn max_len
            src = src[:, :max_len]
            
            # Tạo mask cho src (padding mask)
            encoder_padding_mask = create_padding_mask(src) # Hàm của bạn
            memory_padding_mask_bool = create_standard_padding_mask(src, 3)
            
            # Qua Embedding & Linear
            embeds = self.src_embedding(src)             # (B, S, 3, d_model)
            input_enc = embeds.reshape(B, S, -1)         # (B, S, 3 * d_model)
            input_enc = self.linear(input_enc)           # (B, S, d_model)
            
            # Qua Positional Encoding
            input_enc = self.PE(input_enc)               # (B, S, d_model)
            
            # Lấy Output của Encoder (Memory)
            memory = self.encoder(input_enc, encoder_padding_mask) # (B, S, d_model)

            # ==========================================
            # 2. DECODER PASS (Vòng lặp Autoregressive)
            # ==========================================
            # Khởi tạo target đầu vào bằng token <bos> (Bắt đầu câu)
            # Giả sử self.vocab.bos_idx là index của <bos>
            bos_idx = self.vocab.bos_idx 
            eos_idx = self.vocab.eos_idx
            
            # trg_indices: (B, 1)
            trg_indices = torch.full((B, 1), bos_idx, device=device, dtype=torch.long)
            
            for step in range(max_len):
                S_trg = trg_indices.shape[1]
                
                # Tạo mask cho Decoder
                decoder_causal_mask = create_causal_mask(S_trg, device)
                decoder_padding_mask = create_standard_padding_mask(trg_indices, 3) # Hoặc truyền pad_idx thích hợp
                
                # Qua Embedding & Positional Encoding
                trg_embeds = self.tgt_embedding(trg_indices) # (B, S_trg, d_model)
                dec_input = self.PE(trg_embeds)              # (B, S_trg, d_model)
                
                # Truyền qua Decoder
                logits = self.decoder(
                    dec_input, 
                    memory, 
                    decoder_causal_mask, 
                    decoder_padding_mask, 
                    memory_padding_mask_bool
                ) # (B, S_trg, d_model)
                
                # Tính toán xác suất từ vựng
                out = self.out(logits) # (B, S_trg, vocab_size)
                
                # Chỉ lấy dự đoán của token cuối cùng (tại bước hiện tại)
                next_word_logits = out[:, -1, :] # (B, vocab_size)
                
                # Dùng argmax để chọn từ có xác suất cao nhất (Greedy Decoding)
                next_word = next_word_logits.argmax(dim=-1).unsqueeze(1) # (B, 1)
                
                # Ghép từ vừa dự đoán vào chuỗi target để chạy vòng lặp tiếp theo
                trg_indices = torch.cat([trg_indices, next_word], dim=1)
                
                # Tối ưu: Nếu tất cả các câu trong batch đều đã dự đoán ra <eos>, ta dừng sớm
                if (trg_indices == eos_idx).any(dim=-1).all():
                    break
                    
        return trg_indices
import torch
from torch import nn
import torch.nn.functional as F
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE
from models.transformer_pointer.utils.clone import clones
from models.transformer_pointer.utils.padding_mask import create_padding_mask
from models.transformer_pointer.utils.causal_mask import create_causal_mask
from models.transformer_pointer.blocks.decoder_block import TransformerDecoderBlock
from models.transformer_pointer.blocks.ocr_encoder import OCREncoder
from models.transformer_pointer.embedding.positional_embedding import PositionalEncoding
from models.transformer_pointer.utils.create_context_vector import create_context_vector

class P_gen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prj = nn.Linear(config.d_model * 3, 1)
    def forward(self, x_t, s_t, h_t): 
        combined = torch.cat([x_t, s_t, h_t], dim=-1) # (B, S_trg, 3*d_model)
        scores = self.prj(combined) # scores: (B, S_trg, 1)
        return F.sigmoid(scores)

@META_ARCHITECTURE.register()
class TransformerPointer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.d_model = config.d_model
        self.MAX_LENGTH = vocab.max_sentence_length + 2 # 2 for <bos> + <eos>
        self.config = config
        self.src_max_len = 100
        self.trg_max_len = 100
        
        # Encoder 
        "encoder input: (B, S_src)"
        self.src_embedding = nn.Embedding(self.vocab.vocab_size, self.d_model)
        self.encoder = OCREncoder(self.config.encoder, self.src_embedding)

        # Positional Encoding
        self.PE = PositionalEncoding(self.d_model, 5000)

        # Decoder  
        "decoder input: (B, S_trg, d_model)"
        self.tgt_embedding = nn.Embedding(self.vocab.vocab_size, self.d_model)
        self.decoder = TransformerDecoderBlock(self.config.decoder, self.vocab)
        self.vocab_dist_out = nn.Linear(self.config.d_model, vocab.vocab_size)
        
        # Context_vector
        """
        Returns: 
        h_t* (B, S_trg, d_model)
        """
        
        # Pointer generator class
        """
        Input:
        x_t: decoder input embedded: (B, S_trg, d_model)
        s_t: decoder output: (B, S_trg, d_model)
        h_t: context vector: (B, S_trg, d_model)
        
        Returns: 
        p_gen: (B, S_trg, 1)
        """
        if config.p_gen:
            self.p_generator = P_gen(self.config)
                
        
    def forward(self, src, trg, extended_source_idx, extra_zeros):
        
        max_src_len = 1024
        max_trg_len = 1024
        
        # Nếu câu nguồn dài hơn mức cho phép -> Cắt cụt phần đuôi
        if src.size(1) > max_src_len:
            src = src[:, :max_src_len]
            # BẮT BUỘC phải cắt extended_source_idx y hệt như src để giữ đồng bộ vị trí OOV
            extended_source_idx = extended_source_idx[:, :max_src_len]
            
        # Nếu câu đích dài hơn mức cho phép -> Cắt cụt phần đuôi
        if trg.size(1) > max_trg_len:
            trg = trg[:, :max_trg_len]
        # ========================================================
        
        
        B, S_src = src.shape
        # src: (B, S_src)
        # trg: (B, S_trg)
        
        target = trg[:, 1:]
        # target: (B, S_trg - 1) [0, 1, 2, ... <eos>]
        
        input_for_src = src.clone()
        input_for_src[input_for_src >= self.vocab.vocab_size] = 3
        
        "Encoder"
        encoder_padding_mask = create_padding_mask(src, 0)
        # encoder_padding_mask: (B, S_src)
        encoder_padding_mask = encoder_padding_mask.unsqueeze(1).unsqueeze(1)
        
        assert encoder_padding_mask.ndim == 4, f"Expected 4D tensor but got {encoder_padding_mask.ndim}D"
        memory, _ = self.encoder(input_for_src, encoder_padding_mask, self.PE)
        
        "Decoder"
        # Initial required params
        decoder_input = trg[:, :-1] # (<bos>, 0 , 1, 2, 3, ...) 
        input_for_decoder = decoder_input.clone()
        input_for_decoder[input_for_decoder >= self.vocab.vocab_size] = 3
        _, S_trg = decoder_input.shape
        decoder_padding_mask = create_padding_mask(decoder_input, 0) # decoder_padding: (B, S_trg, d_model)
        decoder_causal_mask = create_causal_mask(S_trg, self.config.device)
        encoder_padding_mask = encoder_padding_mask.squeeze(1).squeeze(1)
        
        # Embedding decoder_input
        embeds = self.tgt_embedding(input_for_decoder)
        # Positional embeds
        x = self.PE(embeds)
        # x: (B, S_trg, d_model)

        decoder_output, decoder_attn_weights = self.decoder(x, memory, decoder_causal_mask, \
                         decoder_padding_mask, encoder_padding_mask)
        # decoder_output: (B, S_trg, d_model)
        # decoder_attn_weights: (B, S_trg, S_src)
        
        p_final = self.vocab_dist_out(decoder_output) # vocab_dist: (B, S_trg, vocab_size)
        p_vocab = F.softmax(p_final, dim=-1)
        
        if self.config.p_gen:
            B, S_trg, _ = p_vocab.shape
            
            """
            POINTER GENERATE
            """
            context_vector = create_context_vector(memory, decoder_attn_weights)
            p_gen = self.p_generator(x_t = embeds, s_t = decoder_output, h_t = context_vector)
            
            """
            FINAL VOCAB DISTRIBUTION
            """
            # p_vocab: (B, S_trg, vocab_size)
            # p_gen: (B, S_trg, 1)
            # extra_zeros: (B, max_oovs)
            # extended_source_idx: (B, max_oovs)
            # decoder_attn_weights: (B, S_trg, S_src)
            
            extra_zeros = extra_zeros.unsqueeze(1)
            extra_zeros = extra_zeros.expand(-1, S_trg, -1)
            # extra_zeros: (B, S_trg, max_oovs)
            
            extended_vocab_dist = torch.cat(((p_vocab * p_gen), extra_zeros), dim=-1) 
            # extended_vocab_dist: (B, S_trg, vocab_size + max_oovs)
            
            attn_dist_ = decoder_attn_weights * (1 - p_gen) # (B, S_trg, S_src)
            max_vocab_limit = extended_vocab_dist.size(-1) - 1
            safe_extended_idx = torch.clamp(extended_source_idx, max=max_vocab_limit)
            
            # Sử dụng safe_extended_idx thay vì extended_source_idx gốc
            index = safe_extended_idx.unsqueeze(1).expand(-1, S_trg, -1)
            
            vocab_dist = extended_vocab_dist.scatter_add(dim = 2, index = index, src=attn_dist_)
            
            if self.config.use_coverage:
                # decoder_attn_weights: (B, S_trg, S_src)
                coverage = torch.cumsum(decoder_attn_weights, dim=1) - decoder_attn_weights
                step_coverage_loss = torch.min(decoder_attn_weights, coverage)
                coverage_loss = step_coverage_loss.sum(dim=-1) 
            else:
                coverage_loss = None
        else:
            vocab_dist = p_vocab
            coverage_loss = None
        
        """
        LOSS
        """
        
        log_probs = torch.log(vocab_dist + 1e-9)
        target_ids = trg[:, 1:]
        
        max_prob_idx = log_probs.size(-1) - 1
        
        # 1. Ép tất cả target_ids không được vượt trần (max) và không được âm (min)
        safe_target_ids = torch.clamp(target_ids, min=0, max=max_prob_idx)
        
        # Tính loss với target an toàn
        nll_loss = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), 
                              safe_target_ids.reshape(-1), 
                              ignore_index=0)
        
        total_loss = nll_loss
        
        if coverage_loss is not None:
            mask = (target_ids != 0).float()
            mask_sum = mask.sum()
            avg_coverage_loss = (coverage_loss * mask).sum() / (mask_sum if mask_sum > 0 else 1.0)
            total_loss = nll_loss + self.config.lambda_cov * avg_coverage_loss

        return None, total_loss
    
    def predict(self, src, extended_source_idx, extra_zeros):
        """
        Hàm suy luận dùng Greedy Search cho mô hình Pointer-Generator.
        
        Args:
            src: (B, S_src) - Câu đầu vào (chứa ID trong vocab)
            extended_source_idx: (B, S_src) - Câu đầu vào (chứa cả ID OOV > vocab_size)
            extra_zeros: (B, max_oovs) - Tensor 0 để đệm cho các từ OOV
            
        Returns:
            outputs: (B, max_length) - Các ID được dự đoán (bao gồm cả OOV)
        """
        self.eval()
        
        # 🛡️ GIÁP 1: Cắt chuỗi đầu vào (Giống hệt hàm forward)
        max_src_len = self.src_max_len 
        if src.size(1) > max_src_len:
            src = src[:, :max_src_len]
            extended_source_idx = extended_source_idx[:, :max_src_len]
            
        B, S_src = src.shape
        device = self.config.device
        
        # 1. Encoder Pass (Chỉ chạy 1 lần)
        # Ép OOV trong src về UNK trước khi qua Encoder
        input_for_src = src.clone()
        input_for_src[input_for_src >= self.vocab.vocab_size] = self.vocab.unk_idx
        
        encoder_padding_mask = create_padding_mask(src, 0)
        enc_mask_4d = encoder_padding_mask.unsqueeze(1).unsqueeze(1)
        memory, _ = self.encoder(input_for_src, enc_mask_4d, self.PE)
        
        # 2. Khởi tạo Decoder Input với token <BOS>
        decoder_input = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=device)
        
        outputs = []
        
        # Cờ đánh dấu các câu đã dịch xong (hữu ích khi B > 1)
        is_finished = torch.zeros(B, dtype=torch.bool, device=device)

        # 3. Vòng lặp giải mã tự hồi quy
        for step in range(self.MAX_LENGTH):
            S_trg = decoder_input.size(1)
            
            # 3.1. Chuẩn bị mask
            decoder_padding_mask = create_padding_mask(decoder_input, 0)
            decoder_causal_mask = create_causal_mask(S_trg, device)
            
            # XỬ LÝ OOV: Tránh lỗi Index Out of Bounds khi đưa qua lớp Embedding
            dec_input_for_embed = decoder_input.clone()
            dec_input_for_embed[dec_input_for_embed >= self.vocab.vocab_size] = self.vocab.unk_idx
            
            # 3.2. Decoder Pass
            embeds = self.tgt_embedding(dec_input_for_embed)
            x = self.PE(embeds)
            
            enc_mask_4d = enc_mask_4d.squeeze(1).squeeze(1)
            
            decoder_output, decoder_attn_weights = self.decoder(
                x, memory, decoder_causal_mask, decoder_padding_mask, enc_mask_4d
            )
            
            # Lấy thông tin của bước thời gian hiện tại (từ cuối cùng)
            last_dec_out = decoder_output[:, -1:, :]      # (B, 1, d_model)
            last_attn = decoder_attn_weights[:, -1:, :]   # (B, 1, S_src)
            last_embed = embeds[:, -1:, :]                # (B, 1, d_model)
            
            # 3.3. Tính Phân phối xác suất (Vocab Distribution)
            p_logits = self.vocab_dist_out(last_dec_out)
            p_vocab = F.softmax(p_logits, dim=-1)         # (B, 1, vocab_size)
            
            if self.config.p_gen:
                # Tính Context Vector
                context_vector = create_context_vector(memory, last_attn)
                
                # Tính p_gen
                p_gen = self.p_generator(last_embed, last_dec_out, context_vector) # (B, 1, 1)
                
                # Mở rộng Vocab Distribution để chứa OOV
                max_oovs = extra_zeros.shape[-1]
                extra_zeros_step = torch.zeros((B, 1, max_oovs), device=device)
                extended_vocab_dist = torch.cat([(p_vocab * p_gen), extra_zeros_step], dim=-1)
                
                # Tính xác suất copy
                attn_dist_ = last_attn * (1 - p_gen)
                
                # 🛡️ GIÁP 2: Clamp extended_source_idx (Giống hệt hàm forward)
                max_vocab_limit = extended_vocab_dist.size(-1) - 1
                safe_extended_idx = torch.clamp(extended_source_idx, max=max_vocab_limit)
                
                index = safe_extended_idx.unsqueeze(1) # (B, 1, S_src)
                
                # Gộp bằng scatter_add
                vocab_dist = extended_vocab_dist.scatter_add(dim=2, index=index, src=attn_dist_)
            else:
                vocab_dist = p_vocab

            # 3.4. Greedy Search: Chọn ID có xác suất cao nhất
            next_token = vocab_dist.argmax(dim=-1) # (B, 1)
            
            # Nếu câu đã xong (gặp EOS trước đó), ép token tiếp theo thành PAD
            next_token = next_token.masked_fill(is_finished.unsqueeze(1), 0)
            
            outputs.append(next_token)
            
            # Cập nhật trạng thái kết thúc
            is_finished = is_finished | (next_token.squeeze(1) == self.vocab.eos_idx)
            
            # 3.5. Dừng sớm nếu TẤT CẢ các câu trong batch đều đã sinh ra EOS
            if is_finished.all():
                break
                
            # Cập nhật chuỗi dự đoán để chuẩn bị cho bước lặp tiếp theo
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

        # Gom danh sách lại thành tensor 2D: (B, độ_dài_câu)
        if outputs:
            outputs = torch.cat(outputs, dim=1)
        else:
            outputs = torch.empty(B, 0, dtype=torch.long, device=device)
            
        return outputs
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
        embeds = self.tgt_embedding(decoder_input)
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
            index = extended_source_idx.unsqueeze(1).expand(-1, S_trg, -1)
            
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
        
        nll_loss = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), 
                              target_ids.reshape(-1), 
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
        B, S_src = src.shape
        device = self.config.device
        
        
        # 1. Encoder Pass (Tính 1 lần duy nhất)
        encoder_padding_mask = create_padding_mask(src, self.vocab.padding_idx)
        enc_mask_4d = encoder_padding_mask.unsqueeze(1).unsqueeze(1)
        memory, _ = self.encoder(src, enc_mask_4d, self.PE)
        
        # 2. Khởi tạo Decoder Input với <BOS>
        # decoder_input: (B, 1)
        decoder_input = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=device)
        
        # Biến lưu trữ kết quả
        outputs = []
        
        # Biến tích lũy cho Coverage (nếu dùng)
        coverage = torch.zeros((B, 1, S_src), device=device) if self.config.use_coverage else None

        # 3. Vòng lặp giải mã tự hồi quy (Autoregressive Decoding)
        for step in range(self.MAX_LENGTH):
            S_trg = decoder_input.size(1)
            
            # 3.1. Chuẩn bị mask cho phần đã sinh ra
            decoder_padding_mask = create_padding_mask(decoder_input, self.vocab.padding_idx)
            decoder_causal_mask = create_causal_mask(S_trg, device)
            
            # Vì ta có thể đã sinh ra từ OOV ở bước trước, mà từ OOV không có Embedding
            # Ta phải thay thế OOV ID bằng UNK ID trước khi nhúng (Embedding)
            dec_input_for_embed = decoder_input.clone()
            dec_input_for_embed[dec_input_for_embed >= self.vocab.vocab_size] = self.vocab.unk_idx
            
            # 3.2. Decoder Pass
            embeds = self.tgt_embedding(dec_input_for_embed)
            x = self.PE(embeds)
            
            decoder_output, decoder_attn_weights = self.decoder(
                x, memory, decoder_causal_mask, decoder_padding_mask, enc_mask_4d
            )
            
            # Chỉ lấy kết quả của từ CUỐI CÙNG được sinh ra (bước thời gian hiện tại)
            # last_dec_out: (B, 1, d_model)
            last_dec_out = decoder_output[:, -1:, :]
            # last_attn: (B, 1, S_src)
            last_attn = decoder_attn_weights[:, -1:, :]
            # last_embed: (B, 1, d_model)
            last_embed = embeds[:, -1:, :]
            
            # 3.3. Tính Vocab Distribution
            p_logits = self.vocab_dist_out(last_dec_out)
            p_vocab = F.softmax(p_logits, dim=-1) # (B, 1, vocab_size)
            
            if self.config.p_gen:
                # Tính context vector cho bước hiện tại
                context_vector = torch.matmul(last_attn, memory)
                
                # Tính xác suất P_gen
                p_gen = self.p_generator(last_embed, last_dec_out, context_vector) # (B, 1, 1)
                
                # Gộp không gian từ vựng OOV
                max_oovs = extra_zeros.shape[-1]
                extra_zeros_step = torch.zeros((B, 1, max_oovs), device=device)
                
                extended_vocab_dist = torch.cat([(p_vocab * p_gen), extra_zeros_step], dim=-1)
                
                # Cộng xác suất Copy
                attn_dist_ = last_attn * (1 - p_gen)
                index = extended_source_idx.unsqueeze(1) # (B, 1, S_src)
                
                vocab_dist = extended_vocab_dist.scatter_add(dim=2, index=index, src=attn_dist_)
                
                # Phạt Coverage vào final distribution (Inference time penalty)
                # Kỹ thuật: Trừ trực tiếp xác suất nếu Coverage lớn
                if self.config.use_coverage and coverage is not None:
                    # Trọng số phạt (có thể tinh chỉnh)
                    cov_penalty = 1.0 
                    # Trừ bớt xác suất copy dựa trên coverage
                    penalty = cov_penalty * torch.min(last_attn, coverage)
                    
                    # Cập nhật Coverage cho bước TIẾP THEO
                    coverage = coverage + last_attn 
            else:
                vocab_dist = p_vocab

            # 3.4. Chọn từ có xác suất cao nhất (Greedy Search)
            # next_token: (B, 1)
            next_token = vocab_dist.argmax(dim=-1)
            outputs.append(next_token)
            
            # Cập nhật decoder_input cho bước sau
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # 3.5. Kiểm tra điều kiện dừng sớm (Chỉ dùng khi Batch Size = 1)
            if B == 1 and next_token.item() == self.vocab.eos_idx:
                break

        # Gom kết quả lại thành tensor (B, length)
        outputs = torch.cat(outputs, dim=1)
        return outputs
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
        B, S_src = src.shape
        device = self.config.device
        
        # 1. Encoder Pass (Chỉ chạy 1 lần)
        # Lưu ý: Nếu trong hàm forward bạn dùng 0 làm padding_idx thì ở đây cũng nên dùng 0
        encoder_padding_mask = create_padding_mask(src, 0)
        enc_mask_4d = encoder_padding_mask.unsqueeze(1).unsqueeze(1)
        memory, _ = self.encoder(src, enc_mask_4d, self.PE)
        
        # 2. Khởi tạo Decoder Input với token <BOS>
        # decoder_input: (B, 1)
        decoder_input = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=device)
        
        outputs = []

        # 3. Vòng lặp giải mã tự hồi quy (Autoregressive Decoding)
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
                
                # Tính xác suất copy và gộp bằng scatter_add
                attn_dist_ = last_attn * (1 - p_gen)
                index = extended_source_idx.unsqueeze(1) # (B, 1, S_src)
                
                vocab_dist = extended_vocab_dist.scatter_add(dim=2, index=index, src=attn_dist_)
            else:
                vocab_dist = p_vocab

            # 3.4. Greedy Search: Chọn ID có xác suất cao nhất
            next_token = vocab_dist.argmax(dim=-1) # (B, 1)
            outputs.append(next_token)
            
            # Cập nhật chuỗi dự đoán để chuẩn bị cho bước lặp tiếp theo
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # 3.5. Dừng sớm (Early Stopping) nếu đã sinh ra <EOS> và Batch = 1
            if B == 1 and next_token.item() == self.vocab.eos_idx:
                break

        # Gom danh sách lại thành tensor 2D: (B, độ_dài_câu)
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
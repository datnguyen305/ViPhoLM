from models.unilm.blocks.transformer_block import TransformerBlock
from models.unilm.embedding.positional_embedding import PositionalEncoding
from vocabs.unilm_vocab import UniLM_Vocab
from models.unilm.utils.clone import clones
import math
import torch
from torch import nn 
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class UniLM(nn.Module):
    def __init__(self, config, vocab: UniLM_Vocab):
        super().__init__()
        self.d_model = config.d_model
        self.vocab = vocab
        self.config = config
        self.MAX_TARGET_LENGTH = vocab.max_sentence_length + 2 # include <bos>, <eos>
        self.MAX_INPUT_LENGTH = vocab.max_input_length + 3 # include <bos>, 2 <eos>
        
        # Embed
        self.token_emb = nn.Embedding(self.vocab.vocab_size, 300)
        if hasattr(self.vocab, 'embedding_matrix') and self.vocab.embedding_matrix is not None and getattr(self.config, 'use_fasttext', False):
            print("Đang nạp trọng số FastText vào Token Embedding...")
            matrix_tensor = torch.tensor(self.vocab.embedding_matrix, dtype=torch.float32)
            self.token_emb.weight.data.copy_(matrix_tensor)
            self.token_emb.weight.requires_grad = False
        self.token_prj = nn.Linear(300, config.d_model)
        self.segment_emb = nn.Embedding(2, config.d_model)
        self.pos_emb = nn.Embedding(self.MAX_INPUT_LENGTH, config.d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Model
        self.big_block = TransformerBlock(config, vocab=self.vocab)
        self.out = nn.Linear(self.config.d_model, self.vocab.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx)
        
    def forward(self, input, input_type_ids, label, src_len):
        
        "TOKEN_EMB"
        input_embedding = self.token_emb(input)
        # input_embedding: (B, S_total, 300) 
        
        "SEGMENT_EMB"    
        segment_embedding = self.segment_emb(input_type_ids)
        # segment_embedding: (B, S_total, d_model)
        
        "POS_EMB"
        pos_ids = torch.arange(input.size(1), device=input.device).unsqueeze(0)
        pos_embedding = self.pos_emb(pos_ids)
        # pos_embedding: (B, S_total, d_model)
        
        "CREATE MASK"
        B, S_total = input.shape
        attention_mask = self._generate_seq2seq_mask(input, src_len)
        
        "FORWARD PASS"
        embeddings = self.token_prj(input_embedding)  * math.sqrt(self.config.d_model) + segment_embedding + pos_embedding
        embeddings = self.dropout(embeddings)
        logits = self.big_block(embeddings, attention_mask=attention_mask)
        # logits: (B, S_total, d_model)
        
        output = self.out(logits) 
        # output: (B, S_total, vocab_size)
        
        "LOSS"
        # output: (B, S_total, vocab_size)
        # label: (B, S_total)
        if label is not None:
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = label[:, 1:].contiguous()
            # shift_logits: <bos> sentence <eos> sentence
            # shift_labels: pad pad pad pad <eos> sentence

            shift_mask = (input_type_ids[:, 1:] == 1)
            shift_labels = shift_labels.masked_fill(~shift_mask, self.vocab.pad_idx)

            loss = self.loss(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return None, loss
    
    def predict(self, input_ids, input_type_ids, src_len):
        B = input_ids.size(0)
        device = input_ids.device
        generated_ids = input_ids.clone()
        generated_type_ids = input_type_ids.clone()

        for step in range(self.MAX_TARGET_LENGTH):
            attention_mask = self._generate_seq2seq_mask(generated_ids, src_len)
            token_emb = self.token_emb(generated_ids)
            token_emb = self.token_prj(token_emb)

            pos_ids = torch.arange(generated_ids.size(1), device=device).unsqueeze(0)
            pos_emb = self.pos_emb(pos_ids)

            seg_emb = self.segment_emb(generated_type_ids)

            embeddings = token_emb * math.sqrt(self.config.d_model) + pos_emb + seg_emb
            embeddings = self.dropout(embeddings)

 
            hidden = self.big_block(embeddings, attention_mask=attention_mask)
            logits = self.out(hidden)  # (B, S, V)

            next_token_logits = logits[:, -1, :]  # (B, V)
            next_token = torch.argmax(next_token_logits, dim=-1)  # (B,)

            next_token = next_token.unsqueeze(1)  # (B, 1)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            next_type = torch.ones((B, 1), dtype=torch.long, device=device)
            generated_type_ids = torch.cat([generated_type_ids, next_type], dim=1)
            
            if (next_token == self.vocab.eos_idx).all():
                break

        return generated_ids
    
    
        
    def _generate_seq2seq_mask(self, input_ids, src_len):
        """
        Hàm tạo ma trận Attention Mask đặc thù của UniLM cho tác vụ Seq2Seq.
        
        Returns: 
            attention_mask: (B, 1, S_total, S_total)
        """
        B, S_total = input_ids.shape
        
        # 1. Tạo lưới tọa độ i (query/row) và j (key/col)
        idx = torch.arange(S_total, device=input_ids.device)
        i = idx.unsqueeze(1)  # Tọa độ hàng (S_total, 1)
        j = idx.unsqueeze(0)  # Tọa độ cột (1, S_total)
        
        # 2. Đưa tensor src_len về đúng shape (B, 1, 1) để broadcast
        src_len_b = src_len.view(B, 1, 1)
        
        # 3. Khởi tạo Seq2Seq Mask cốt lõi
        seq2seq_mask = (j < src_len_b) | (i >= j)  # Shape: (B, S_total, S_total)
        
        # 4. Padding Mask: Ngăn mô hình chú ý vào các token <pad>
        pad_id = getattr(self.vocab, 'pad_id', 0) 
        pad_mask = (input_ids != pad_id).unsqueeze(1)  # Shape: (B, 1, S_total)
        
        # 5. Kết hợp Mask và thêm chiều cho Multi-Head Attention 
        attention_mask = (seq2seq_mask & pad_mask).unsqueeze(1) # Shape: (B, 1, S_total, S_total)
        
        # 6. Chuyển đổi sang dạng chuẩn cho Transformer PyTorch
        attention_mask = torch.where(attention_mask, 0.0, float('-inf'))
        
        return attention_mask
        
        
         
        
import torch
from torch import nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE
import math

class Phrasal_Lexeme(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size, bias = False)

    def forward(self, query, key):
        # query: (B, S, D)
        # key: (B, S, D)
        Sk = key.size(1)
        Sq = query.size(1)

        if Sq != Sk:
            raise ValueError(
                f"Phrasal_Lexeme module chỉ hỗ trợ ma trận vuông (Self-Attention). "
                f"Nhưng hiện tại nhận được Sq={Sq} và Sk={Sk}. "
                f"Hãy kiểm tra xem bạn có đang vô tình gọi module này trong Cross-Attention của Decoder không."
            )


        intermediate = self.linear(query)  # (B, S, D)

        # intermediate: (B, S, D)
        # key.transpose: (B, D, S)
        scores = torch.matmul(intermediate, key.transpose(-2, -1))  # (B, S, S)
        r_left = torch.diagonal(scores, offset=-1, dim1=-2, dim2=-1) # (B, S-1)
        r_right = torch.diagonal(scores, offset=1, dim1=-2, dim2=-1) # (B, S-1)
        left_side = r_left[:, :-1]
        right_side = r_right[:, 1:] 

        comparison = torch.stack([left_side, right_side], dim=-1) # (B, S-2, 2)
        pr = torch.softmax(comparison, dim=-1) # (B, S-2, 2)
        """
            => pr[:,i,0] left_side
            => pr[:,i,1] right_side
            dim = (B, S-2)

        """
        # Pi = sqrt(left_side * right_side)
        pr_i_right = pr[:, :-1, 1] 
        pr_iplus1_left = pr[:, 1:, 0]
        Pi = torch.sqrt(pr_i_right * pr_iplus1_left + 1e-9)  
        # Pi: (B, S-2)
        log_P = torch.log(Pi + 1e-8)  # tránh log(0)
        # log_P: (B, S-2)
        Pi_j = torch.exp(torch.sum(log_P, dim=-1)) 
        
        # Pi_j: (B)
        return Pi_j
 
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_head
        self.d_model = config.hidden_size
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_model // self.num_heads

        self.linear_q = nn.Linear(self.d_model, self.d_model)
        self.linear_k = nn.Linear(self.d_model, self.d_model)
        self.linear_v = nn.Linear(self.d_model, self.d_model)
        self.linear_out = nn.Linear(self.d_model, self.d_model)

        self.phrasal_lexeme = Phrasal_Lexeme(config)

    def forward(self, query, key, value, mask=None, causal_mask=None):
        B, S_q, _ = query.size() # Độ dài của Query
        S_k = key.size(1)       # Độ dài của Key (quan trọng!)
        S_v = value.size(1)     # Độ dài của Value

        # Linear projections
        Q = self.linear_q(query).reshape(B, S_q, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, S_q, d_k)
        K = self.linear_k(key).reshape(B, S_k, self.num_heads, self.d_k).transpose(1, 2)    # (B, num_heads, S_k, d_k)
        V = self.linear_v(value).reshape(B, S_v, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, S_v, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32, device=Q.device))
        final_mask = None
        if mask is not None and causal_mask is not None:
            # Kết hợp cả hai bằng phép toán AND (logic &)
            # mask (B, 1, 1, S_k) & casual_mask (1, 1, S_q, S_k) -> (B, 1, S_q, S_k)
            final_mask = mask & causal_mask
        elif mask is not None:
            final_mask = mask
        elif causal_mask is not None:
            final_mask = causal_mask

        if final_mask is not None:
            scores = scores.masked_fill(final_mask == 0, float('-inf'))

        # Tính xác suất cụm từ P_{i,j} (B)
        # Lưu ý: Nếu bài báo muốn mỗi head một P, bạn cần sửa Phrasal_Lexeme trả về (B, H)
        p_score = self.phrasal_lexeme(query, key) 
        
        # Chuyển đổi p_score để nhân element-wise (B, H, 1, 1) hoặc (B, 1, 1, 1)
        if p_score.dim() == 1:
            p_gate = p_score.view(B, 1, 1, 1)
        else:
            p_gate = p_score.unsqueeze(-1).unsqueeze(-1)

        attn_weights = torch.softmax(scores, dim=-1)  # (B, num_heads, S_q, S_k)

        attn_weights = attn_weights * p_gate  # (B, num_heads, S_q, S_k)

        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, S_q, S_k) * (B, num_heads, S_k, d_k) = (B, num_heads, S_q, d_k)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).reshape(B, S_q, self.d_model)  # (B, S_q, num_heads * d_k)
        output = self.linear_out(attn_output)  # (B, S_q, d_model)
        return output
        
class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.linear1 = nn.Linear(config.hidden_size, config.ffn_hidden)
        self.linear2 = nn.Linear(config.ffn_hidden, config.hidden_size)
        self.dropout = nn.Dropout(config.drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        # 1. Cơ chế Attention bạn muốn sửa nằm ở đây
        self.multi_head_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        # 2. Các thành phần chuẩn của một lớp Encoder
        self.linear1 = nn.Linear(config.hidden_size, config.dim_feedforward)
        self.dropout = nn.Dropout(config.drop_prob)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.drop_prob)
        self.dropout2 = nn.Dropout(config.drop_prob)

    def forward(self, src, src_mask=None, src_causal_mask=None):
        # 1. Multi-Head Attention
        attn_input = self.norm1(src)
        attn_output = self.dropout1(self.multi_head_attn(attn_input, attn_input, attn_input, mask=src_mask, causal_mask=src_causal_mask))
        # src (B, S, hidden_size)
        output1 = src + attn_output

        # 2. Feed Forward Network
        ffn_output = self.dropout2(self.feed_forward(self.norm2(output1)))
        output = ffn_output + output1
        #  # (B, S_q, hidden-size)
        return output 
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config, vocab) for _ in range(config.n_layers)
        ])

    def forward(self, src, src_mask=None, src_causal_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, src_causal_mask=src_causal_mask)
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        # 1. Cơ chế Attention bạn muốn sửa nằm ở đây
        self.multi_head_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        # 2. Các thành phần chuẩn của một lớp Encoder
        self.linear1 = nn.Linear(config.hidden_size, config.dim_feedforward)
        self.dropout = nn.Dropout(config.drop_prob)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.drop_prob)
        self.dropout2 = nn.Dropout(config.drop_prob)

    def forward(self, src, memory, trg_mask=None, trg_causal_mask=None, src_mask=None):
        # 1. Masked Multi-Head Attention
        attn_input = self.norm1(src)
        attn_output = self.dropout1(self.multi_head_attn(attn_input, attn_input, attn_input, mask=trg_mask, causal_mask=trg_causal_mask))
        attn_input_2 = src + attn_output

        # 2. Multi-Head Attention
        attn_input_2 = self.norm2(attn_input_2)
        attn_output = self.dropout1(self.multi_head_attn(attn_input_2, memory, memory, mask=src_mask))
        # src (B, S, hidden_size)
        ff_input = attn_input_2 + attn_output

        # 3. Feed Forward Network
        ffn_output = self.dropout2(self.feed_forward(self.norm2(ff_input)))
        output = ffn_output + ff_input
        return output

class TransformerDecoderBlock(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config, vocab) for _ in range(config.n_layers)
        ])

    def forward(self, trg, memory, trg_mask=None, trg_causal_mask=None, src_mask=None):
        output = trg
        for layer in self.layers:
            output = layer(output, memory, trg_mask=trg_mask, trg_causal_mask=trg_causal_mask, src_mask=src_mask)
        return output
    
def create_padding_mask(seq, pad_idx):
    # seq: (B, S)
    # mask: (B, S), True tại vị trí từ thật, False tại vị trí <pad>
    mask = (seq != pad_idx) 
    
    # Expand chiều để broadcast với scores (B, H, S, S)
    # Kết quả: (B, 1, 1, S)
    return mask.unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq, device):
    # 1. Tạo ma trận vuông (size x size) toàn số 1
    # 2. torch.tril giữ lại tam giác dưới (triangular lower), xóa tam giác trên thành 0
    mask = torch.tril(torch.ones((seq, seq), device=device))
    
    # 3. Đưa về dạng (1, 1, S, S) để tương thích với scores (B, H, S, S)
    return mask.unsqueeze(0).unsqueeze(0).bool()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [B, L, D]
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)
    
@META_ARCHITECTURE.register()
class Testing(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx=vocab.pad_idx)
        self.output_embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx=vocab.pad_idx)
        self.encoder = TransformerEncoderBlock(config.encoder, vocab)
        self.decoder = TransformerDecoderBlock(config.decoder, vocab)
        self.PE = PositionalEncoding(config.hidden_size, max_len=5000)
        
        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2 # +2 for BOS and EOS tokens
        self.d_model = config.d_model

        self.max_length = config.max_length
        self.loss = nn.CrossEntropyLoss()
        self.fc_out = nn.Linear(config.hidden_size, vocab.vocab_size)

    def forward(self, src, trg):
        # max_length_input = max_length_output 
        src = src[:, :self.max_length]   # src [<bos> ... <eos>] 
        trg = trg[:, :self.max_length]   # trg [ ... <eos>]

        # Đảm bảo có token <eos> ở cuối chuỗi
        if self.vocab.eos_idx not in src:
            src[:, -1] = self.vocab.eos_idx
        if self.vocab.eos_idx not in trg:
            trg[:, -1] = self.vocab.eos_idx
        if self.vocab.bos_idx not in trg:
            trg[:, 0] = self.vocab.bos_idx
        # trg [<bos> ... <eos>]
        
        # Nếu chuỗi ngắn hơn max_length, điền pad_idx vào
        src_padding = self.max_length - src.size(1)
        trg_padding = self.max_length - trg.size(1)
        if src_padding > 0:
            src = torch.cat([src, torch.full((src.size(0), src_padding), self.vocab.pad_idx, dtype=torch.long, device=src.device)], dim=1)
        if trg_padding > 0:
            trg = torch.cat([trg, torch.full((trg.size(0), trg_padding), self.vocab.pad_idx, dtype=torch.long, device=trg.device)], dim=1)

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
        device = src.device
        batch_size = src.size(0)
        
        # 1. Encoder (Chạy 1 lần duy nhất)
        src_emb = self.PE(self.input_embedding(src))
        src_mask = create_padding_mask(src, self.vocab.pad_idx).to(device)
        encoder_outs = self.encoder(src_emb, src_mask=src_mask)
        decoder_input = torch.full((batch_size, 1), self.vocab.bos_idx, dtype=torch.long, device=device)

        # 2. Khởi tạo chuỗi đích với token bắt đầu <bos>
        # trg_indexes: (1, 1)
        outputs = []
        for _ in range(self.MAX_LENGTH):
            # Cần cộng PE cho input của decoder mỗi bước
            trg_emb = self.PE(self.output_embedding(decoder_input))
            
            # Tạo mask cho chuỗi hiện tại
            trg_mask = create_padding_mask(decoder_input, self.vocab.pad_idx)
            trg_causal_mask = create_causal_mask(decoder_input.size(1), device=device)

            # 3. Forward qua Decoder
            outs = self.decoder(trg_emb, encoder_outs, 
                                trg_mask=trg_mask, 
                                trg_causal_mask=trg_causal_mask, 
                                src_mask=src_mask)
            
            # Lấy logit của token cuối cùng: (Batch, 1, Vocab_size)
            logits = self.fc_out(outs[:, -1:, :])
            
            # Dự đoán token tiếp theo: (Batch, 1)
            next_token = logits.argmax(dim=-1)
            
            outputs.append(next_token)
            
            # Cập nhật decoder_input để dự đoán bước tiếp theo (Autoregressive)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # Kiểm tra dừng (chỉ áp dụng nếu Batch_size = 1 hoặc xử lý riêng lẻ)
            if batch_size == 1 and next_token.item() == self.vocab.eos_idx:
                break

        # 4. Nối các token dự đoán lại thành Tensor (Batch, Seq_len)
        # Giống hệt logic outputs = torch.cat(outputs, dim=1) của bạn
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
import torch
from torch import nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class MultiHeadAttention(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.num_heads = config.nhead
        self.d_model = config.hidden_size
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_model // self.num_heads

        self.linear_q = nn.Linear(self.d_model, self.d_model)
        self.linear_k = nn.Linear(self.d_model, self.d_model)
        self.linear_v = nn.Linear(self.d_model, self.d_model)
        self.linear_out = nn.Linear(self.d_model, self.d_model)
    def forward(self, query, key, value, mask=None, causal_mask=None):
        B, S, _ = query.size()

        # Linear projections
        Q = self.linear_q(query).reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, S, d_k)
        K = self.linear_k(key).reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)    # (B, num_heads, S, d_k)
        V = self.linear_v(value).reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, S, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32, device=Q.device))
        final_mask = None
        if mask is not None and causal_mask is not None:
            # Kết hợp cả hai bằng phép toán AND (logic &)
            # mask (B, 1, 1, S_k) & casual_mask (1, 1, S, S_k) -> (B, 1, S, S_k)
            final_mask = mask & causal_mask
        elif mask is not None:
            final_mask = mask
        elif causal_mask is not None:
            final_mask = causal_mask

        if final_mask is not None:
            scores = scores.masked_fill(final_mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)  # (B, num_heads, S, S)

        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, S, S) * (B, num_heads, S, d_k) = (B, num_heads, S, d_k)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).reshape(B, S, self.d_model)  # (B, S, d_model)
        output = self.linear_out(attn_output)  # (B, S, d_model)

        return output
        
class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

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
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, src, src_mask=None, src_causal_mask=None):
        # 1. Multi-Head Attention
        attn_input = self.norm1(src)
        attn_output = self.dropout1(self.multi_head_attn(attn_input, attn_input, attn_input, mask=src_mask, causal_mask=src_causal_mask))
        # src (B, S, hidden_size)
        output1 = src + attn_output

        # 2. Feed Forward Network
        ffn_output = self.dropout2(self.feed_forward(self.norm2(output1)))
        output = ffn_output + output1
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
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, src, memory, src_mask=None, src_causal_mask=None):
        # 1. Masked Multi-Head Attention
        attn_input = self.norm1(src)
        attn_output = self.dropout1(self.multi_head_attn(attn_input, attn_input, attn_input, mask=src_mask, causal_mask=src_causal_mask))
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

@META_ARCHITECTURE.register()
class Testing(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.encoder = TransformerEncoderBlock(config.encoder, vocab)
        self.decoder = TransformerDecoderBlock(config.decoder, vocab)
        
        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2 # +2 for BOS and EOS tokens
        self.d_model = config.d_model

        self.loss = nn.CrossEntropyLoss()
        self.fc_out = nn.Linear(config.hidden_size, vocab.vocab_size)

    def forward(self, src, trg):
        src_mask = create_padding_mask(src, self.vocab.pad_idx)
        trg_mask = create_padding_mask(trg, self.vocab.pad_idx)
        trg_causal_mask = create_causal_mask(trg.size(1), device=trg.device)

        encoder_outs = self.encoder(src, src_mask=src_mask, src_causal_mask=None)
    
        outs = self.decoder(trg, encoder_outs, trg_mask=trg_mask, trg_causal_mask=trg_causal_mask, src_mask=src_mask)
        # outs: (B, S, d_model)
        logits = self.fc_out(outs) # (B, S, vocab_size)
        V = self.vocab.vocab_size
        loss = self.loss(logits.view(-1, V), trg.view(-1))
        return logits, loss
    
    
    def predict(self, src):
        # 1. Chuẩn bị đầu vào cho Encoder
        # src: (1, S_src) - Giả sử bạn truyền vào 1 câu đơn lẻ
        src_mask = create_padding_mask(src, self.vocab.pad_idx).to(src.device)
        encoder_outs = self.encoder(src, src_mask=src_mask, src_causal_mask=None)


        # 2. Khởi tạo chuỗi đích với token bắt đầu <bos>
        # trg_indexes: (1, 1)
        trg_indexes = [self.vocab.bos_idx]
        for _ in range(self.MAX_LENGTH):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(src.device)  # (1, len)

            trg_mask = create_padding_mask(trg_tensor, self.vocab.pad_idx).to(src.device)
            trg_causal_mask = create_causal_mask(trg_tensor.size(1), device=src.device)

            # 3. Dự đoán token tiếp theo
            outs = self.decoder(trg_tensor, encoder_outs, trg_mask=trg_mask, trg_causal_mask=trg_causal_mask, src_mask=src_mask)
            logits = self.fc_out(outs)  # (1, len, vocab_size)

            next_token = logits[:, -1, :].argmax(dim=-1).item()
            trg_indexes.append(next_token)

            # Dừng nếu gặp token <eos>
            if next_token == self.vocab.eos_idx:
                break
        return trg_indexes
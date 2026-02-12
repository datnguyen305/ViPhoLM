import torch
from torch import nn
import torch.nn.functional as F
from vocabs.hierarchy_vocab import Hierarchy_Vocab
from builders.model_builder import META_ARCHITECTURE
import numpy as np 
import math
import copy

class StandardEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.d_model, config.head, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

    def forward(self, x, mask):
        # mask ở đây là padding mask cho các câu (B, S)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, key_padding_mask=mask)[0])
        return self.sublayer[1](x, self.feed_forward)

class StandardTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([StandardEncoderLayer(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

"""
Oke
"""
class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = config.d_model
        self.d_q = config.d_kv
        self.d_kv = config.d_kv
        self.head = config.head

        self.fc_q = nn.Linear(config.d_model, config.head * config.d_kv)
        self.fc_k = nn.Linear(config.d_model, config.head * config.d_kv)
        self.fc_v = nn.Linear(config.d_model, config.head * config.d_kv)

    def forward(self, queries, keys, values, group_prob, attention_mask):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.head, self.d_q).permute(0, 2, 1, 3)   # (b_s, h, nq, d_q)
        k = self.fc_k(keys).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 3, 1)     # (b_s, h, nk, d_kv)
        v = self.fc_v(values).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 1, 3)   # (b_s, h, nk, d_kv)

        att = torch.matmul(q, k) / np.sqrt(self.d_kv)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            # attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            att.masked_fill(attention_mask == 0, -1e4)
        att = torch.softmax(att, dim=-1)
        att = att * group_prob
        output = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b_s, -1, self.d_model)

        return output

"""
Oke
"""   

class GroupAttention(nn.Module):
    def __init__(self, config):
        super(GroupAttention, self).__init__()
        self.h = config.head
        self.d_k = config.d_model // config.head
        self.linear_key = nn.Linear(self.d_k, self.d_k)
        self.linear_query = nn.Linear(self.d_k, self.d_k)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, context, eos_mask, prior):
        bs, seq_len = context.size()[:2]

        context = self.norm(context).view(bs, seq_len, self.h, self.d_k).transpose(1, 2)

        a = torch.diag(torch.ones(seq_len - 1), 1).long().to(context.device)
        b = torch.diag(torch.ones(seq_len), 0).long().to(context.device)
        c = torch.diag(torch.ones(seq_len - 1), -1).long().to(context.device)

        mask = torch.logical_and(eos_mask, (a+c))
        
        key = self.linear_key(context)
        query = self.linear_query(context)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k
        
        scores = scores.masked_fill(mask == 0, -1e4)
        neibor_attn = F.softmax(scores, dim = -1)
        neibor_attn = torch.sqrt(neibor_attn*neibor_attn.transpose(-2,-1) + 1e-4)
        neibor_attn = prior + (1. - prior)*neibor_attn

        tri_matrix = torch.triu(torch.ones(seq_len, seq_len), diagonal = 0).float().to(context.device)
        tri_matrix = tri_matrix.unsqueeze(0).unsqueeze(0)
        t = torch.log(neibor_attn + 1e-9).masked_fill(a == 0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b == 0, 1e-4)
        
        return g_attn, neibor_attn

"""
Oke
"""

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

"""
Oke
"""

class PositionwiseFeedForward(nn.Module):
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
    
"""
Oke
"""

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

"""
Oke
"""
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, config):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


"""
In progress
"""
class EncoderLayer(nn.Module):
    def __init__(self, config, vocab: Hierarchy_Vocab):
        super().__init__()
        # Attention
        self.group_attn = GroupAttention(config)
        self.self_attn = ScaledDotProductAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)
        self.size = config.d_model

    def forward(self, x, mask, group_prob):
        group_prob, break_prob = self.group_attn(x, mask, group_prob)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob, mask))
        
        return self.sublayer[1](x, self.feed_forward), group_prob, break_prob
    

class DecoderLayer(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        # Self-Attention chuẩn (thay thế GroupAttention + ScaledDotProduct)
        self.self_attn = nn.MultiheadAttention(config.d_model, config.head, batch_first=True)
        
        # Cross-Attention (Giữ nguyên)
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.head, batch_first=True)
        
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 3)

    def forward(self, x, memory, tgt_mask, memory_mask, group_prob=None):
        """
        x: (B, T, D)
        memory: (B, S, D)
        tgt_mask: (T, T) - Causal mask chuẩn cho nn.MultiheadAttention
        memory_mask: (B, S) - Padding mask cho encoder
        """
        # 1. Standard Self-Attention
        # Lưu ý: tgt_mask ở đây nên là dạng (T, T) hoặc (B*H, T, T) cho nn.MultiheadAttention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, attn_mask=tgt_mask)[0])
        
        # 2. Cross-Attention
        x = self.sublayer[1](x, lambda x: self.cross_attn(query=x, 
                                                         key=memory, 
                                                         value=memory, 
                                                         key_padding_mask=~memory_mask.bool())[0])
        # 3. Feed Forward
        x = self.sublayer[2](x, self.feed_forward)
        
        return x, None, None # Trả về None cho group_prob và break_prob


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config, vocab: Hierarchy_Vocab):
        super().__init__()
        self.word_embed = nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=vocab.pad_idx)
        self.layers = clones(EncoderLayer(config, vocab), 3)
        self.norm = nn.LayerNorm(config.d_model)

        self.pos_embed = PositionalEncoding(config.d_model)

    def forward(self, inputs, mask):
        x = self.word_embed(inputs)
        x = self.pos_embed(x)

        break_probs = []
        group_prob = 0.

        for layer in self.layers:
            x, group_prob, break_prob = layer(x, mask,group_prob)
            break_probs.append(break_prob)

        x = self.norm(x)
        break_probs = torch.stack(break_probs, dim=1)

        return x, break_probs

class TransformerDecoderBlock(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.word_embed = nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=vocab.pad_idx)
        self.pos_embed = PositionalEncoding(config.d_model)
        self.layers = clones(DecoderLayer(config, vocab), config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, trg, memory, tgt_mask, memory_mask):
        """
        trg: (B, T) - Target token IDs
        memory: (B, S, D) - Encoder sentence representations
        tgt_mask: (B, T) - Target mask
        memory_mask: (B, S) - Encoder sentence mask
        """
        x = self.word_embed(trg)
        x = self.pos_embed(x)
        
        group_prob = 0.
        break_probs = []
        
        for layer in self.layers:
            x, group_prob, break_prob = layer(x, memory, tgt_mask, memory_mask, group_prob)
            break_probs.append(break_prob)
            
        return self.norm(x), torch.stack(break_probs, dim=1)


@META_ARCHITECTURE.register()
class EnhancedAttnTransformerModel(nn.Module):
    def __init__(self, config, vocab: Hierarchy_Vocab):
        super().__init__()
        self.vocab = vocab
        self.d_model = config.d_model
        self.MAX_LENGTH = vocab.max_sentence_length + 2
        self.config = config

        # Khối Encoder & Decoder
        self.word_encoder = TransformerEncoderBlock(config.encoder, vocab)
        self.sentence_encoder = StandardTransformerEncoder(config.encoder)
        self.decoder = TransformerDecoderBlock(config.decoder, vocab)
        
        # Positional Encoding
        self.Word_PE = PositionalEncoding(self.d_model, max_len=5000)
        self.Sen_PE = PositionalEncoding(self.d_model, max_len=100)
        
        self.fc_out = nn.Linear(self.d_model, vocab.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def forward(self, src, trg):
        B, S, W = src.size()
        device = src.device

        # --- PHASE 1: WORD-LEVEL ENCODER ---
        src_flat = src.view(B * S, W)
        # Sử dụng hàm create_padding_mask của bạn (B*S, 1, 1, W)
        src_mask_flat = create_padding_mask(src_flat, self.vocab.pad_idx).to(device)
        
        # Word Encoder trả về (B*S, W, D)
        encoder_outs_word, _ = self.word_encoder(src_flat, src_mask_flat)
        
        # --- PHASE 2: SENTENCE-LEVEL ENCODER ---
        # Pooling về (B, S, D)
        sent_repr = encoder_outs_word.view(B, S, W, -1).mean(dim=2)
        sent_repr = self.Sen_PE(sent_repr)
        
        # Xác định câu nào là padding (B, S)
        # sent_mask: 1 tại câu thật, 0 tại câu padding
        sent_mask = (src.sum(dim=-1) != self.vocab.pad_idx * W).float().to(device)
        
        # nn.MultiheadAttention cần True tại vị trí Padding (sent_mask == 0)
        memory = self.sentence_encoder(sent_repr, (sent_mask == 0))

        # --- PHASE 3: DECODER ---
        trg_input = trg[:, :-1]
        
        # Tạo kết hợp Mask cho Decoder Self-Attention
        # 1. Padding mask cho target: (B, 1, 1, T)
        trg_pad_mask = create_padding_mask(trg_input, self.vocab.pad_idx).to(device)
        # 2. Causal mask cho target: (1, 1, T, T)
        trg_causal_mask = create_causal_mask(trg_input.size(1), device).to(device)
        
        # Kết hợp: Không nhìn vào <pad> AND không nhìn vào tương lai
        full_trg_mask = trg_pad_mask & trg_causal_mask

        # Forward qua Decoder (Memory_mask là sent_mask)
        decoder_outs, _ = self.decoder(trg_input, memory, full_trg_mask, sent_mask)
        
        # Tính Loss
        logits = self.fc_out(decoder_outs)
        loss = self.loss(logits.view(-1, logits.size(-1)), trg[:, 1:].contiguous().view(-1))
        
        return logits, loss

    def predict(self, src, max_len=None):
        device = self.config.device
        B, S, W = src.size()
        max_len = max_len if max_len is not None else self.MAX_LENGTH

        # 1. Encoder (Giống forward)
        src_flat = src.view(B * S, W)
        src_mask_flat = create_padding_mask(src_flat, self.vocab.pad_idx).to(device)
        encoder_outs_word, _ = self.word_encoder(src_flat, src_mask_flat)
        
        sent_repr = encoder_outs_word.view(B, S, W, -1).mean(dim=2)
        sent_repr = self.Sen_PE(sent_repr)
        sent_mask = (src.sum(dim=-1) != self.vocab.pad_idx * W).float().to(device)
        memory = self.sentence_encoder(sent_repr, (sent_mask == 0))

        # 2. Decode tự hồi quy
        ys = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=device)

        for i in range(max_len):
            # Cập nhật mask cho độ dài hiện tại của ys
            trg_pad_mask = create_padding_mask(ys, self.vocab.pad_idx).to(device)
            trg_causal_mask = create_causal_mask(ys.size(1), device).to(device)
            full_trg_mask = trg_pad_mask & trg_causal_mask

            # Decoder forward
            out, _ = self.decoder(ys, memory, full_trg_mask, sent_mask)
            
            # Dự đoán token kế tiếp
            logits = self.fc_out(out[:, -1, :])
            next_word = torch.argmax(logits, dim=-1, keepdim=True)
            
            ys = torch.cat([ys, next_word], dim=1)
            if next_word.item() == self.vocab.eos_idx:
                break

        return ys
    


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
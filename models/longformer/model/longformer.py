import torch
from torch import nn
import torch.nn.functional as F
import math

# ==========================================
# 1. IMPORTS TỪ PROJECT CỦA BẠN
# ==========================================
from vocabs.viword_vocab import Vocab
from builders.model_builder import META_ARCHITECTURE
from models.longformer.utils.padding_mask import create_padding_mask, create_standard_padding_mask
from models.longformer.utils.causal_mask import create_causal_mask

# CHÚ Ý: ĐÂY LÀ NƠI BẠN IMPORT LONGFORMER ATTENTION CỦA BẠN
# Thay đổi đường dẫn import này cho đúng với project của bạn nhé!
from models.longformer.layers.longformer_self_attention import LongformerSelfAttention 


# ==========================================
# 2. CÁC LỚP CƠ BẢN (Positional Encoding & FFN)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, S, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LongformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LongformerSelfAttention(config)
        self.feed_forward = FeedForward(config)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, padding_mask, global_attention_mask):
        attn_output = self.self_attn(
            hidden_states=x,
            attention_mask=padding_mask,
            global_attention_mask=global_attention_mask
        )
        
        # Nếu attention của bạn trả về tuple (output, weights), lấy phần tử đầu
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class LongformerEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([LongformerEncoderLayer(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, padding_mask, global_attention_mask):
        for layer in self.layers:
            x = layer(x, padding_mask, global_attention_mask)
        return self.norm(x)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.d_model, config.nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.nhead, batch_first=True)
        
        self.feed_forward = FeedForward(config)
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, memory, causal_mask, tgt_padding_mask, memory_padding_mask):
        # 1. Causal Self-Attention
        attn1, _ = self.self_attn(
            x, x, x, 
            attn_mask=causal_mask, 
            key_padding_mask=tgt_padding_mask
        )
        x = x + self.dropout(attn1)
        x = self.norm1(x)

        # 2. Cross-Attention
        attn2, _ = self.cross_attn(
            x, memory, memory, 
            key_padding_mask=memory_padding_mask
        )
        x = x + self.dropout(attn2)
        x = self.norm2(x)

        # 3. Feed Forward
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, memory, causal_mask, tgt_padding_mask, memory_padding_mask):
        for layer in self.layers:
            x = layer(x, memory, causal_mask, tgt_padding_mask, memory_padding_mask)
        return self.norm(x)

@META_ARCHITECTURE.register()
class Longformer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.config = config
        self.d_model = config.d_model
        self.MAX_LENGTH = config.inference_length

        self.PE = PositionalEncoding(self.d_model, dropout=config.dropout, max_len=self.config.max_len + 10)

        self.src_embedding = nn.Embedding(vocab.vocab_size, config.d_model)
        self.tgt_embedding = nn.Embedding(vocab.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.encoder = LongformerEncoderBlock(config)
        self.decoder = TransformerDecoderBlock(config)
        
        self.outs = nn.Linear(config.d_model, vocab.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)                                  
    def forward(self, src, trg):
        B, S_src = src.shape
        src = src[:, :self.config.max_len]
        trg = trg[:, :self.config.max_len]

        # Padding
        if src.shape[1] < self.config.max_len:
            pad = torch.full((B, self.config.max_len - src.shape[1]), self.vocab.pad_idx, device=src.device, dtype=torch.long)
            src = torch.cat([src, pad], dim=1)

        if trg.shape[1] < self.config.max_len:
            pad = torch.full((B, self.config.max_len - trg.shape[1]), self.vocab.pad_idx, device=trg.device, dtype=torch.long)
            trg = torch.cat([trg, pad], dim=1)

        # Tính toán Masks cho Encoder
        encoder_padding_mask = create_padding_mask(src, self.vocab.pad_idx)
        
        # Bật Global Attention cho token đầu tiên (BOS)
        global_attention_mask = torch.zeros_like(src, device=src.device, dtype=torch.long)
        global_attention_mask[:, 0] = 1 

        target = trg[:, 1:]        
        decoder_input = trg[:, :-1] 

        # --- Encoder ---
        x_src = self.dropout(self.src_embedding(src))
        x_src = self.PE(x_src)
        memory = self.encoder(x_src, encoder_padding_mask, global_attention_mask)

        # --- Decoder ---
        decoder_padding_mask = create_standard_padding_mask(decoder_input, self.vocab.pad_idx)
        decoder_causal_mask = create_causal_mask(decoder_input.size(1), self.config.device)
        memory_padding_mask_bool = create_standard_padding_mask(src, self.vocab.pad_idx)

        x_tgt = self.dropout(self.tgt_embedding(decoder_input))
        x_tgt = self.PE(x_tgt)

        dec_out = self.decoder(x_tgt, memory, decoder_causal_mask, decoder_padding_mask, memory_padding_mask_bool)
        
        logits = self.outs(dec_out)
        total_loss = self.loss_fn(logits.view(-1, self.vocab.vocab_size), target.reshape(-1))

        return 0, total_loss 

    def predict(self, src):
        B, S = src.shape
        src = src[:, :self.config.max_len]

        if src.shape[1] < self.config.max_len:
            pad = torch.full((B, self.config.max_len - src.shape[1]), self.vocab.pad_idx, device=src.device, dtype=torch.long)
            src = torch.cat([src, pad], dim=1)

        encoder_padding_mask = create_padding_mask(src, self.vocab.pad_idx)
        memory_padding_mask_bool = create_standard_padding_mask(src, self.vocab.pad_idx)
        
        global_attention_mask = torch.zeros_like(src, device=src.device, dtype=torch.long)
        global_attention_mask[:, 0] = 1 

        # --- Encoder ---
        x_src = self.dropout(self.src_embedding(src))
        x_src = self.PE(x_src)
        memory = self.encoder(x_src, encoder_padding_mask, global_attention_mask)

        # --- Decoder Tự Hồi Quy (Autoregressive) ---
        decoder_input = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=self.config.device)
        outputs = []
        
        for _ in range(self.MAX_LENGTH): 
            x_tgt = self.dropout(self.tgt_embedding(decoder_input))
            x_tgt = self.PE(x_tgt)

            trg_mask = create_standard_padding_mask(decoder_input, self.vocab.pad_idx)
            trg_causal_mask = create_causal_mask(decoder_input.size(1), self.config.device)

            dec_out = self.decoder(x_tgt, memory, trg_causal_mask, trg_mask, memory_padding_mask_bool)
            
            last_token_out = dec_out[:, -1:, :] 
            logits = self.outs(last_token_out) 
            next_token = logits.argmax(dim=-1) 
            
            outputs.append(next_token)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if B == 1 and next_token.item() == self.vocab.eos_idx:
                break
                
        outputs = torch.cat(outputs, dim=1)
        return outputs
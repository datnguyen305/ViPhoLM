import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * out + self.beta

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class ScaleDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, head, length, d_tensor = k.size()
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))

        score = self.softmax(score)
        score = torch.nan_to_num(score, nan=0.0)  # Handle NaN
        score = self.dropout(score)
        v = score @ v
        return v, score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.n_head = n_head
        self.attention = ScaleDotProductAttention(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class HeposMultiHeadAttention(nn.Module):
    """
    HEPOS: Head-wise Positional Strides Attention
    Paper: "Efficient Attentions for Long Document Summarization"
    """
    def __init__(self, d_model, n_head, stride=None, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.d_model = d_model
        self.n_head = n_head
        self.stride = stride if stride is not None else n_head
        self.head_dim = d_model // n_head
        
        assert self.stride > 0, "stride must be positive"
        assert self.stride <= n_head, f"stride ({self.stride}) should not exceed n_head ({n_head})"
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Tq, _ = q.size()
        B, Tk, _ = k.size()
        D = self.head_dim

        # Linear projections and reshape
        Q = self.w_q(q).view(B, Tq, self.n_head, D).transpose(1, 2)  # (B, H, Tq, D)
        K = self.w_k(k).view(B, Tk, self.n_head, D).transpose(1, 2)  # (B, H, Tk, D)
        V = self.w_v(v).view(B, Tk, self.n_head, D).transpose(1, 2)  # (B, H, Tk, D)

        # Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # (B, H, Tq, Tk)

        # =====================================
        # HEPOS: Vectorized head-wise stride mask
        # =====================================
        device = scores.device
        indices = torch.arange(Tk, device=device)  # [Tk]
        head_offsets = torch.arange(self.n_head, device=device)  # [H]
        
        # Each head h attends to positions where (i - h) % stride == 0
        # Broadcasting: [H, 1] and [1, Tk] -> [H, Tk]
        hepos_mask = ((indices.unsqueeze(0) - head_offsets.unsqueeze(1)) % self.stride != 0)
        hepos_mask = hepos_mask.unsqueeze(0).unsqueeze(2)  # [1, H, 1, Tk]
        
        scores = scores.masked_fill(hepos_mask, float('-inf'))

        # Padding mask (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle NaN from all -inf rows
        attn = self.dropout(attn)

        # Attention output
        out = torch.matmul(attn, V)  # (B, H, Tq, D)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        
        return self.w_o(out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, attn_type='full'):
        super().__init__()
        
        # Choose attention type
        if attn_type == 'full':
            self.attention = MultiHeadAttention(d_model, n_head, drop_prob)
        # elif attn_type == 'sliding_window':
        #     self.attention = SlidingWindowAttention(...)
        # elif attn_type == 'lsh':
        #     self.attention = LSHAttention(...)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")
        
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, src_mask):
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, stride=None):
        super().__init__()
        
        # Self-attention (causal)
        self.self_attention = MultiHeadAttention(d_model, n_head, drop_prob)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        # Cross-attention with HEPOS
        self.enc_dec_attention = HeposMultiHeadAttention(d_model, n_head, stride, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        # FFN
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # Self-attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # Cross-attention (HEPOS)
        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # FFN
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1), :]

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model, padding_idx=1)

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(drop_prob)

    def forward(self, x):
        tok = self.tok_emb(x)
        pos = self.pos_emb(x)
        return self.drop_out(tok + pos)

@dataclass
class EncoderConfig:
    d_model: int = 512
    n_layers: int = 6
    n_head: int = 8
    ffn_hidden: int = 2048
    drop_prob: float = 0.1
    max_len: int = 5000
    attn_type: str = 'full'  # 'full', 'sliding_window', 'lsh', 'sinkhorn'
    device: str = 'cuda'

@dataclass
class DecoderConfig:
    d_model: int = 512
    n_layers: int = 6
    n_head: int = 8
    ffn_hidden: int = 2048
    drop_prob: float = 0.1
    max_len: int = 1024
    stride: Optional[int] = None  # None = use n_head
    device: str = 'cuda'

class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig, vocab, use_checkpoint=False):
        super().__init__()
        self.emb = TransformerEmbedding(
            vocab_size=vocab.vocab_size,
            d_model=config.d_model,
            max_len=config.max_len,
            drop_prob=config.drop_prob,
            device=config.device
        )
        
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=config.d_model,
                ffn_hidden=config.ffn_hidden,
                n_head=config.n_head,
                drop_prob=config.drop_prob,
                attn_type=config.attn_type
            ) for _ in range(config.n_layers)
        ])
        
        self.use_checkpoint = use_checkpoint

    def forward(self, x, src_mask):
        x = self.emb(x)
        
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(layer, x, src_mask)
            else:
                x = layer(x, src_mask)
        
        return x

class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig, vocab):
        super().__init__()
        self.emb = TransformerEmbedding(
            vocab_size=vocab.vocab_size,
            d_model=config.d_model,
            max_len=config.max_len,
            drop_prob=config.drop_prob,
            device=config.device
        )
        
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=config.d_model,
                ffn_hidden=config.ffn_hidden,
                n_head=config.n_head,
                drop_prob=config.drop_prob,
                stride=config.stride
            ) for _ in range(config.n_layers)
        ])
        
        self.linear = nn.Linear(config.d_model, vocab.vocab_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)
        
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.linear(trg)
        return output

@META_ARCHITECTURE.register()
class TransformerHeposModel(nn.Module):
    """
    Transformer with HEPOS (Head-wise Positional Strides) attention
    for efficient long document processing
    """
    def __init__(self, encoder_config: EncoderConfig, decoder_config: DecoderConfig, vocab: Vocab):
        super().__init__()
        self.src_pad_idx = vocab.pad_idx
        self.trg_pad_idx = vocab.pad_idx
        self.trg_bos_idx = vocab.bos_idx
        self.trg_eos_idx = vocab.eos_idx
        self.vocab_size = len(vocab)
        
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        
        self.encoder = Encoder(encoder_config, vocab)
        self.decoder = Decoder(decoder_config, vocab)
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)

    def forward(self, src, trg):
        # Truncate if needed
        src = src[:, :self.encoder_config.max_len]
        trg = trg[:, :self.decoder_config.max_len]

        # Shift right for decoder input
        B = trg.size(0)
        bos = torch.full((B, 1), self.trg_bos_idx, device=trg.device)
        dec_in = torch.cat([bos, trg[:, :-1]], dim=1)

        # Create masks
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(dec_in)

        # Forward pass
        enc = self.encoder(src, src_mask)
        out = self.decoder(dec_in, enc, trg_mask, src_mask)

        # Calculate loss
        loss = self.loss_fn(out.reshape(-1, out.size(-1)), trg.reshape(-1))

        return out, loss

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

    def make_trg_mask(self, trg):
        pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        T = trg.size(1)
        causal = torch.tril(torch.ones((T, T), device=trg.device)).bool()
        return pad_mask & causal  # (B, 1, T, T)

    @torch.no_grad()
    def predict(self, src, max_len=None):
        self.eval()
        
        max_len = max_len or self.decoder_config.max_len
        src = src[:, :self.encoder_config.max_len]
        
        src_mask = self.make_src_mask(src)
        enc = self.encoder(src, src_mask)

        B = src.size(0)
        dec = torch.full((B, 1), self.trg_bos_idx, device=src.device)
        outputs = []

        for _ in range(max_len):
            trg_mask = self.make_trg_mask(dec)
            out = self.decoder(dec, enc, trg_mask, src_mask)
            
            next_tok = out[:, -1].argmax(dim=-1, keepdim=True)
            outputs.append(next_tok)
            
            dec = torch.cat([dec, next_tok], dim=1)
            
            if (next_tok == self.trg_eos_idx).all():
                break

        return torch.cat(outputs, dim=1)
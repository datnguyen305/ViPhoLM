import torch
import torch.nn as nn
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE

"""
def to_cuda(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor
"""

class HATConfig:
    def __init__(self, **kwargs):
        self.segment_len = kwargs.get("segment_len", 128)
        self.hidden_size = kwargs.get("hidden_size", 256)
        self.max_segments = kwargs.get("max_segments", 10)
        self.nhead = kwargs.get("nhead", 4)
        self.num_layers = kwargs.get("num_layers", 4)
        self.encoder_layout = kwargs.get(
            "encoder_layout",
            {str(i): {"sentence_encoder": True, "document_encoder": (i > 0)} for i in range(kwargs.get("num_layers", 4))}
        )


class HATEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, segment_len, max_segments):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.token_pos_emb = nn.Embedding(segment_len + 1, hidden_size)
        self.segment_pos_emb = nn.Embedding(max_segments, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        B, S, T = input_ids.size()
        token_pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).unsqueeze(0)
        segment_pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        segment_pos_ids = segment_pos_ids.clamp(max=self.segment_pos_emb.num_embeddings - 1)


        token_emb = self.token_emb(input_ids)
        pos_emb = self.token_pos_emb(token_pos_ids)
        token_emb = token_emb + pos_emb

        seg_cls = token_emb[:, :, 0, :].clone()
        seg_cls = seg_cls + self.segment_pos_emb(segment_pos_ids)

        return self.dropout(token_emb), self.dropout(seg_cls)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, nhead, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=1)

    def forward(self, x):
        return self.encoder(x)


class HATLayer(nn.Module):
    def __init__(self, hidden_size, nhead, dropout=0.1, use_swe=True, use_cse=True):
        super().__init__()
        self.use_swe = use_swe
        self.use_cse = use_cse

        if use_swe:
            self.swe = TransformerLayer(hidden_size, nhead, dropout)
        if use_cse:
            self.cse = TransformerLayer(hidden_size, nhead, dropout)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_seq, segment_cls, S, T):
        B, L, H = token_seq.size()

        if self.use_swe:
            token_seq = self.swe(token_seq)

        if self.use_cse:
            segment_cls = self.cse(segment_cls)
            seg_ctx = segment_cls.unsqueeze(2).expand(-1, -1, T, -1).reshape(B, -1, H)
            token_seq = token_seq + seg_ctx[:, :token_seq.size(1), :]

        return self.norm(self.dropout(token_seq)), segment_cls


class HATEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead, encoder_layout):
        super().__init__()
        self.layers = nn.ModuleList([
            HATLayer(hidden_size, nhead,
                     use_swe=encoder_layout[str(i)]['sentence_encoder'],
                     use_cse=encoder_layout[str(i)]['document_encoder'])
            for i in range(num_layers)
        ])

    def forward(self, token_seq, segment_cls, S, T):
        for layer in self.layers:
            token_seq, segment_cls = layer(token_seq, segment_cls, S, T)
        return token_seq


@META_ARCHITECTURE.register()
class HATModel(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        if not isinstance(config, HATConfig):
            config = HATConfig(**config.__dict__)

        self.segment_len = config.segment_len
        self.hidden_size = config.hidden_size
        self.d_model = self.hidden_size
        self.max_segments = config.max_segments
        self.nhead = config.nhead
        self.num_layers = config.num_layers
        self.encoder_layout = config.encoder_layout

        self.vocab_size = len(vocab)
        self.embeddings = HATEmbeddings(self.vocab_size, self.hidden_size, self.segment_len, self.max_segments)
        self.encoder = HATEncoder(self.num_layers, self.hidden_size, self.nhead, self.encoder_layout)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        S = L // self.segment_len
        input_ids = input_ids[:, :S * self.segment_len].contiguous().view(B, S, self.segment_len)

        cls_token = torch.full((B, S, 1), fill_value=1, dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([cls_token, input_ids], dim=2)  # [B, S, T+1]
        #truncate input_ids to ensure it does not exceed max_len
        max_len = self.segment_len * self.max_segments # Ensure max_len is set correctly
        input_ids = input_ids[:, :max_len] # đảm bảo không bao giờ tạo L > segment_len * max_segments.


        x, seg_cls = self.embeddings(input_ids)
        x = x.view(B, S * (self.segment_len + 1), self.hidden_size)

        x = self.encoder(x, seg_cls, S, self.segment_len + 1)
        x = x.view(B, S, self.segment_len + 1, -1)[:, :, 1:, :].reshape(B, -1, self.hidden_size)
        logits = self.lm_head(x)

        loss = None
        # start debug 
        if labels is not None:
            # Lấy chiều dài chuỗi mà model đã xử lý và tạo ra logits
            processed_seq_len = logits.size(1)
            original_label_len = labels.size(1)

            # Kiểm tra xem labels có ngắn hơn logits không (chắc chắn là có)
            if original_label_len < processed_seq_len:
                # Tính toán số lượng cần đệm thêm
                padding_size = processed_seq_len - original_label_len
                
                # Tạo một tensor đệm toàn số 0.
                # Dùng giá trị 0 vì F.cross_entropy có ignore_index=0
                padding = torch.zeros(
                    (labels.size(0), padding_size), 
                    dtype=torch.long, 
                    device=labels.device
                )
                
                # Nối labels gốc với phần đệm để chúng có cùng chiều dài
                labels = torch.cat([labels, padding], dim=1)
            
            # Đề phòng trường hợp hiếm labels dài hơn, ta cắt bớt
            elif original_label_len > processed_seq_len:
                labels = labels[:, :processed_seq_len]

            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.contiguous().view(-1)
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0)

        return logits, loss

    def predict(self, input_ids):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_ids)
            return torch.argmax(logits, dim=-1)

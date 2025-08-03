import torch
import torch.nn as nn
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE

def to_cuda(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor

@META_ARCHITECTURE.register()
class HATModel(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.vocab_size = len(vocab)
        self.segment_len = getattr(config, 'segment_len', 128)
        self.hidden_size = getattr(config, 'hidden_size', 256)
        self.max_segments = getattr(config, 'max_segments', 10)
        self.nhead = getattr(config, 'nhead', 4)
        self.num_layers_swe = getattr(config, 'num_layers_swe', 2)
        self.num_layers_cse = getattr(config, 'num_layers_cse', 2)
        self.encoder_layout = getattr(config, 'encoder_layout', {
            str(i): {"sentence_encoder": True, "document_encoder": (i > 0)} for i in range(self.num_layers_swe)
        })

        self.d_model = self.hidden_size  # For warmup scheduler

        self.token_pos_emb = nn.Embedding(self.segment_len + 1, self.hidden_size)  # +1 for [CLS]
        self.segment_pos_emb = nn.Embedding(self.max_segments, self.hidden_size)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        # Layers per layout config
        self.layers = nn.ModuleList()
        for i in range(len(self.encoder_layout)):
            layer_cfg = self.encoder_layout[str(i)]
            self.layers.append(HATLayer(self.hidden_size, self.nhead,
                                        use_swe=layer_cfg['sentence_encoder'],
                                        use_cse=layer_cfg['document_encoder']))

        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        S = L // self.segment_len
        input_ids = input_ids.view(B, S, self.segment_len)

        cls_token = torch.full((B, S, 1), fill_value=1, dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([cls_token, input_ids], dim=2)  # [B, S, T+1]
        token_pos_ids = torch.arange(self.segment_len + 1, device=input_ids.device).unsqueeze(0).unsqueeze(0)
        token_pos_emb = self.token_pos_emb(token_pos_ids)  # [1, 1, T+1, H]

        seg_reps = []
        segment_outputs = []
        for i in range(S):
            seg = input_ids[:, i, :]  # [B, T+1]
            seg_emb = self.embedding(seg) + token_pos_emb
            segment_outputs.append(seg_emb)
            seg_reps.append(seg_emb[:, 0, :])  # [CLS]

        x = torch.cat(segment_outputs, dim=1)  # [B, S*(T+1), H]
        seg_cls = torch.stack(seg_reps, dim=1)  # [B, S, H]

        # Pos embedding for segments
        seg_pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        seg_cls = seg_cls + self.segment_pos_emb(seg_pos_ids)

        # Apply layers
        for layer in self.layers:
            x, seg_cls = layer(x, seg_cls, S, self.segment_len + 1)

        # Get token outputs (remove CLS tokens)
        x = x.view(B, S, self.segment_len + 1, -1)[:, :, 1:, :].contiguous().view(B, -1, self.hidden_size)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
        return logits, loss

    def predict(self, input_ids):
        with torch.no_grad():
            logits, _ = self.forward(input_ids)
            return torch.argmax(logits, dim=-1)


class HATLayer(nn.Module):
    def __init__(self, hidden_size, nhead, use_swe=True, use_cse=True):
        super().__init__()
        self.use_swe = use_swe
        self.use_cse = use_cse

        if self.use_swe:
            self.swe = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True), num_layers=1
            )
        if self.use_cse:
            self.cse = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True), num_layers=1
            )

    def forward(self, token_seq, segment_cls, S, T):
        B = token_seq.size(0)

        if self.use_swe:
            token_seq = self.swe(token_seq)  # [B, S*(T+1), H]

        if self.use_cse:
            segment_cls = self.cse(segment_cls)  # [B, S, H]
            seg_ctx = segment_cls.unsqueeze(2).repeat(1, 1, T - 1, 1).view(B, -1, segment_cls.size(-1))
            token_seq = token_seq + seg_ctx[:, :token_seq.size(1), :]

        return token_seq, segment_cls

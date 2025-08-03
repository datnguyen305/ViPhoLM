import torch
import torch.nn as nn
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab

def to_cuda(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor

class EntityCNNEncoder(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=1)

    def forward(self, entity_seq):
        x = entity_seq.transpose(1, 2)
        x = F.relu(self.conv(x))
        return x.transpose(1, 2)

class SentenceEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)

    def forward(self, sent_emb):
        x = sent_emb.transpose(1, 2)
        x = F.relu(self.conv(x)).transpose(1, 2)
        outputs, _ = self.bilstm(x)
        return outputs

class ContentSelector(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lstm = nn.LSTM(d_model * 2, d_model, batch_first=True)
        self.attn_entity = nn.Linear(d_model + d_model, 1)
        self.attn_sent = nn.Linear(d_model + d_model * 2, 1)
        self.prob_proj = nn.Linear(d_model * 3, 1)

    def forward(self, sents, entities):
        batch, n_sent, _ = sents.size()
        h_t = torch.zeros(1, batch, sents.size(-1)).to(sents.device)
        c_t = torch.zeros(1, batch, sents.size(-1)).to(sents.device)
        selected = []

        for _ in range(3):
            ent_ctx = self.attend(entities, h_t[-1], self.attn_entity)
            sent_ctx = self.attend(sents, h_t[-1], self.attn_sent)
            context = torch.cat([h_t[-1], ent_ctx, sent_ctx], dim=-1)
            p = torch.sigmoid(self.prob_proj(context))
            top_idx = torch.topk(p.squeeze(-1), 1)[1]
            selected_sent = sents[torch.arange(batch), top_idx]
            lstm_input = torch.cat([selected_sent, ent_ctx], dim=-1).unsqueeze(1)
            _, (h_t, c_t) = self.lstm(lstm_input, (h_t, c_t))
            selected.append(selected_sent.unsqueeze(1))

        return torch.cat(selected, dim=1)

    def attend(self, memory, query, attn_layer):
        query = query.unsqueeze(1).expand(-1, memory.size(1), -1)
        scores = attn_layer(torch.cat([query, memory], dim=-1)).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), memory).squeeze(1)

class AbstractGenerator(nn.Module):
    def __init__(self, d_model, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.decoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.copy_attn = nn.Linear(d_model * 2, 1)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, selected, tgt=None, max_len=50, teacher_forcing=False):
        batch = selected.size(0)
        hidden = None
        outputs = []
        input_token = torch.zeros(batch, 1, dtype=torch.long, device=selected.device)

        for t in range(max_len):
            emb = self.embedding(input_token)
            dec_out, hidden = self.decoder(emb, hidden)
            copy_score = self.copy_attn(torch.cat([dec_out, selected.mean(1, keepdim=True)], dim=-1))
            vocab_logits = self.out_proj(dec_out)
            final_logits = vocab_logits + copy_score
            outputs.append(final_logits)
            input_token = tgt[:, t].unsqueeze(1) if teacher_forcing and tgt is not None else final_logits.argmax(-1)

        return torch.cat(outputs, dim=1)

@META_ARCHITECTURE.register()
class SENCASummarizer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.d_model = config.d_model
        self.padding_idx = config.padding_idx
        self.max_tgt_len = config.max_tgt_len
        self.teacher_forcing = config.teacher_forcing

        self.embedding = nn.Embedding(len(vocab), self.d_model, padding_idx=self.padding_idx)
        self.entity_emb = nn.Embedding(config.num_entity_types, self.d_model)

        self.entity_encoder = EntityCNNEncoder(self.d_model)
        self.sent_encoder = SentenceEncoder(self.d_model)
        self.selector = ContentSelector(self.d_model)
        self.generator = AbstractGenerator(self.d_model, len(vocab), self.padding_idx)

    def forward(self, src, src_entities, tgt=None):
        src_emb = self.embedding(src)
        ent_emb = self.entity_emb(src_entities)
        ent_repr = self.entity_encoder(ent_emb)
        sent_repr = self.sent_encoder(src_emb)
        selected = self.selector(sent_repr, ent_repr)
        logits = self.generator(selected, tgt, max_len=self.max_tgt_len, teacher_forcing=self.teacher_forcing and tgt is not None)
        return logits
    def predict(self, src, src_entities, max_len=None):
        """
        Inference mode: autoregressive decoding without teacher forcing.
        """
        self.eval()
        max_len = max_len or self.max_tgt_len
        with torch.no_grad():
            src_emb = self.embedding(src)
            ent_emb = self.entity_emb(src_entities)
            ent_repr = self.entity_encoder(ent_emb)
            sent_repr = self.sent_encoder(src_emb)
            selected = self.selector(sent_repr, ent_repr)
            logits = self.generator(selected, tgt=None, max_len=max_len, teacher_forcing=False)
            output_ids = logits.argmax(dim=-1)  # (batch, seq_len)
        return output_ids

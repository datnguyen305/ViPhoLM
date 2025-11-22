import torch
from torch import nn
from .components import ConvEncoder, LSTMPointerNet

class EntityAwareExtractor(nn.Module):
    def __init__(self, vocab_size, emb_dim, conv_hidden, lstm_hidden, lstm_layer, bidirectional, n_hop, dropout):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Sentence Encoder
        self.sent_conv_encoder = ConvEncoder(emb_dim, conv_hidden, [3,4,5], dropout)
        self.sent_lstm_encoder = nn.LSTM(
            3 * conv_hidden, lstm_hidden, lstm_layer, 
            bidirectional=bidirectional, dropout=dropout, batch_first=True
        )

        # Entity Encoder
        self.entity_conv_encoder = ConvEncoder(emb_dim, conv_hidden, [2,3,4], dropout)
        
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        entity_dim = 3 * conv_hidden

        self.pointer_decoder = LSTMPointerNet(
            enc_out_dim, lstm_hidden, lstm_layer, dropout, n_hop, entity_dim
        )
        
    def forward(self, article_sents, sent_nums, clusters, cluster_nums, target_indices):
        # 1. Mã hóa câu
        sents_emb = self._embedding(article_sents) # [B, max_sents, max_words, D_emb]
        B, NS, NW, D = sents_emb.shape
        sents_emb_flat = sents_emb.view(B * NS, NW, D)
        conv_sents = self.sent_conv_encoder(sents_emb_flat)
        # Lấy vector đại diện cho câu (ví dụ: trung bình)
        sent_vectors = conv_sents.mean(dim=1).view(B, NS, -1)
        
        packed_sents = nn.utils.rnn.pack_padded_sequence(sent_vectors, sent_nums, batch_first=True, enforce_sorted=False)
        sent_mem, _ = self.sent_lstm_encoder(packed_sents)
        sent_mem, _ = nn.utils.rnn.pad_packed_sequence(sent_mem, batch_first=True)
        
        # 2. Mã hóa thực thể
        entities_emb = self._embedding(clusters) # [B, max_ents, max_mentions, D_emb]
        B, NE, NM, D = entities_emb.shape
        entities_emb_flat = entities_emb.view(B * NE, NM, D)
        conv_ents = self.entity_conv_encoder(entities_emb_flat)
        entity_mem = conv_ents.mean(dim=1).view(B, NE, -1)

        # 3. Chuẩn bị đầu vào cho Pointer Decoder
        bs, nt = target_indices.size()
        d = sent_mem.size(2)
        ptr_in = torch.gather(
            sent_mem, dim=1, index=target_indices.unsqueeze(2).expand(bs, nt, d)
        )
        
        # 4. Chạy Pointer Decoder
        scores = self.pointer_decoder(sent_mem, entity_mem, ptr_in, sent_nums, cluster_nums)
        return scores
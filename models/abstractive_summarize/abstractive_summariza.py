import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.hierachy_vocab import Hierachy_Vocab

class HierarchicalFeatureRichEncoder(nn.Module):
    def __init__(self, config, vocab: Hierachy_Vocab, tfidf_size=10):
        super().__init__()
        self.word_emb = nn.Embedding(vocab.vocab_size, config.emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(vocab.pos_size, config.pos_dim, padding_idx=0)
        self.ner_emb = nn.Embedding(vocab.ner_size, config.ner_dim, padding_idx=0)
        self.tfidf_emb = nn.Embedding(tfidf_size, config.tfidf_dim, padding_idx=0)
        
        self.total_emb_dim = config.emb_dim + config.pos_dim + config.ner_dim + config.tfidf_dim

        # --- Word-level RNN and Attention --- 
        self.word_rnn = nn.GRU(self.total_emb_dim, config.hidden_size, 
                                batch_first=True, bidirectional=True)
        self.word_attention = nn.Linear(config.hidden_size * 2, 1)

        # --- Sentence-level RNN and Attention --- 
        self.sent_pos_emb = nn.Embedding(100, config.hidden_size * 2)
        self.sent_rnn = nn.GRU(config.hidden_size * 2, config.hidden_size, 
                                batch_first=True, bidirectional=True)
        self.sent_attention = nn.Linear(config.hidden_size * 2, 1)
        self.dropout = nn.Dropout(config.dropout)

        def forward(self, input_ids, pos_ids, ner_ids, tfidf_ids):
            # input_ids shape: (B, S, W)
            B, S, W = input_ids.size()
            # Nhúng và nối đặc trưng
            combined_emb = torch.cat([
                self.word_emb(input_ids),
                self.pos_emb(pos_ids),
                self.ner_emb(ner_ids),
                self.tfidf_emb(tfidf_ids)
            ], dim=-1) # (B, S, W, Total_Emb)

            combined_emb = self.dropout(combined_emb)

            # --- Cấp độ Từ (Word-level) ---
            flat_emb = combined_emb.reshape(B * S, W, -1) # (B*S, W, Total_Emb)
            word_hiddens, _ = self.word_rnn(flat_emb) # (B*S, W, H*2)
            
            word_attn_scores = self.word_attention(word_hiddens).squeeze(-1) # (B*S, W)
            # WORD_ATTENTION
            word_attn_weights = F.softmax(word_attn_scores, dim=-1).unsqueeze(1) # (B*S, 1, W)


            # (B*S, 1, W) * (B*S, W, H*2) = (B*S, 1, H*2) -> squeeze(1) -> (B*S, H*2)
            sent_vectors = torch.bmm(word_attn_weights, word_hiddens).squeeze(1) # (B*S, H*2)
            sent_vectors = sent_vectors.reshape(B, S, -1) # (B, S, H*2)

            # --- Cấp độ Câu (Sentence-level) ---
            # (S) -> (1,S) -> (B,S)
            pos_indices = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S) # Có thể dùng PE của transformer
            sent_vectors = sent_vectors + self.sent_pos_emb(pos_indices)
            
            sent_hiddens, _ = self.sent_rnn(sent_vectors) # (B, S, H*2)
            sent_attn_scores = self.sent_attention(sent_hiddens).squeeze(-1) # (B, S)
            # SENTENCE_ATTENTION
            sent_attn_weights = F.softmax(sent_attn_scores, dim=-1) # (B, S)

            # --- Re-scaled  word-level attention --- 
            word_attn_weights = word_attn_weights.reshape(B, S, W) # (B, S, W)
            reweighted_word_attn = word_attn_weights * sent_attn_weights.unsqueeze(-1) # (B, S, W) * (B, S, 1) = (B, S, W)
            rescaled_flat = reweighted_word_attn.reshape(B, -1)
            eps = 1e-10 # Tránh chia cho 0
            total_sum = rescaled_flat.sum(dim=-1, keepdim=True) + eps
            normalized_weights = rescaled_flat / total_sum # (B, S*W)
            final_attn_weights = normalized_weights.view(B, S, W)

            # Context vector tổng thể
            # (B, S, W, H*2) * (B, S, W, 1) -> sum -> (B, H*2)
            word_hiddens = word_hiddens.view(B, S, W, -1)
            context_vector = torch.sum(final_attn_weights.unsqueeze(-1) * word_hiddens, dim=[1, 2])

            return word_hiddens, sent_hiddens, final_attn_weights, context_vector

class PointerGeneratorDecoder(nn.Module):
    def












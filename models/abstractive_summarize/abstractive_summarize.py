import torch
import torch.nn as nn
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE
# Đảm bảo đường dẫn import đúng với project của bạn
from vocabs.hierachy_vocab import Hierachy_Vocab 

# --- CÁC MODULE ATTENTION & UTILS ---

class WordLevelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SỬA LỖI: Bỏ việc nhân 2. Input từ Encoder đã là 512, khớp với hidden_size config
        # Dùng 'hidden_size' cho khớp với file config YAML của bạn
        hidden_dim = config.hidden_size 
        
        self.W1 = nn.Linear(hidden_dim, hidden_dim) # Nhận 512 -> Ra 512
        self.W2 = nn.Linear(hidden_dim, hidden_dim) # Nhận 512 -> Ra 512
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, word_inputs, decoder_hidden, B, S):
        # word_inputs: (B*S, W, H) -> (20800, W, 512)
        # decoder_hidden: (B, H)
        
        # Chuẩn bị decoder hidden: (B, H) -> (B, S, H) -> (B*S, 1, H)
        dec_hid = decoder_hidden.unsqueeze(1).expand(-1, S, -1).reshape(B * S, 1, -1)

        # Tính score
        # W1(word_inputs) + W2(dec_hid)
        scores = self.V(torch.tanh(
            self.W1(word_inputs) + self.W2(dec_hid)
        )) # (B*S, W, 1)
        
        alphas = F.softmax(scores, dim=1) # (B*S, W, 1)

        # Context: (B*S, 1, W) * (B*S, W, H) -> (B*S, 1, H)
        context_per_sent = torch.bmm(alphas.transpose(1, 2), word_inputs)
        
        # Mean pooling về cấp câu: (B, S, H)
        context_word = context_per_sent.view(B, S, -1).mean(dim=1, keepdim=True)
        
        return context_word, alphas.view(B, S, -1) 

class SentenceLevelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SỬA LỖI TƯƠNG TỰ
        hidden_dim = config.hidden_size
        
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, sent_inputs, decoder_hidden):
        # sent_inputs: (B, S, H)
        # decoder_hidden: (B, H) -> (B, 1, H)
        dec_hid = decoder_hidden.unsqueeze(1) 
        
        scores = self.V(torch.tanh(
            self.W1(sent_inputs) + self.W2(dec_hid)
        )) # (B, S, 1)

        alphas = F.softmax(scores, dim=1) # (B, S, 1)
        
        # Context
        context_sent = torch.bmm(alphas.transpose(1, 2), sent_inputs)

        return context_sent, alphas.squeeze(-1)

def rescale_attention(p_a_w, p_a_s):
    # p_a_w: (B, S, W), p_a_s: (B, S)
    numerator = p_a_w * p_a_s.unsqueeze(-1) # (B, S, W)
    
    B, S, W = numerator.size()
    flat_numerator = numerator.view(B, -1) # (B, Nd)
    
    denominator = flat_numerator.sum(dim=-1, keepdim=True) + 1e-10
    p_a = flat_numerator / denominator
    return p_a

def calculate_loss(p_final, p_selection, target_idx, vocab_size):
    # target_idx: (B)
    g_i = (target_idx < vocab_size).float() # 1 nếu in-vocab, 0 nếu OOV

    # Lấy xác suất tại đúng index target
    p_target = p_final.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    p_target = p_target + 1e-12 

    # Loss hybrid
    term_in_vocab = g_i * torch.log(p_target * p_selection.squeeze())
    term_oov = (1 - g_i) * torch.log(p_target * (1 - p_selection.squeeze()))

    loss = -(term_in_vocab + term_oov)
    return loss.mean()

# --- ENCODER ---

class HierarchicalFeatureRichEncoder(nn.Module):
    def __init__(self, config, vocab: Hierachy_Vocab, tfidf_size=10):
        super().__init__()
        self.word_emb = nn.Embedding(vocab.vocab_size, config.emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(vocab.pos_size, config.pos_dim, padding_idx=0)
        self.ner_emb = nn.Embedding(vocab.ner_size, config.ner_dim, padding_idx=0)
        self.tfidf_emb = nn.Embedding(tfidf_size, config.tfidf_dim, padding_idx=0)
        
        self.total_emb_dim = config.emb_dim + config.pos_dim + config.ner_dim + config.tfidf_dim

        # Word Level
        self.word_rnn = nn.GRU(self.total_emb_dim, config.hidden_size, 
                                batch_first=True, bidirectional=True)
        self.word_attention = nn.Linear(config.hidden_size * 2, 1)

        # Sentence Level
        self.sent_pos_emb = nn.Embedding(100, config.hidden_size * 2)
        self.sent_rnn = nn.GRU(config.hidden_size * 2, config.hidden_size, 
                                batch_first=True, bidirectional=True)
        self.sent_attention = nn.Linear(config.hidden_size * 2, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, pos_ids, ner_ids, tfidf_ids):
        B, S, W = input_ids.size()
        
        # --- BẢO VỆ TUYỆT ĐỐI INPUT IDS ---
        # Đảm bảo mọi ID đều nằm trong khoảng [0, vocab_size - 1]
        # self.word_emb.num_embeddings chính là vocab_size
        vocab_limit = self.word_emb.num_embeddings
        
        # Tạo bản sao để không ảnh hưởng luồng Pointer
        safe_input_ids = input_ids.clone()
        
        # Thay thế các giá trị không hợp lệ bằng UNK (thường là index 2 hoặc 3)
        # unk_idx = 2 (Giả sử, bạn có thể lấy từ config nếu có)
        unk_idx = 2 
        
        # Mask lỗi quá lớn
        mask_overflow = safe_input_ids >= vocab_limit
        if mask_overflow.any():
            safe_input_ids[mask_overflow] = unk_idx
            
        # Mask lỗi số âm (nếu có lỗi lạ từ dataset)
        mask_underflow = safe_input_ids < 0
        if mask_underflow.any():
            safe_input_ids[mask_underflow] = unk_idx

        # 1. Embedding với input đã an toàn
        combined_emb = torch.cat([
            self.word_emb(safe_input_ids), # Dùng safe_input_ids
            self.pos_emb(pos_ids),
            self.ner_emb(ner_ids),
            self.tfidf_emb(tfidf_ids)
        ], dim=-1)
        combined_emb = self.dropout(combined_emb)

        # 2. Word RNN (Flatten B*S)
        flat_emb = combined_emb.reshape(B * S, W, -1)
        
        # Lấy word_last_hidden (2, B*S, H) để khởi tạo Decoder sau này
        word_hiddens, word_last_hidden = self.word_rnn(flat_emb) 
        
        # 3. Word Attention
        word_attn_scores = self.word_attention(word_hiddens).squeeze(-1)
        word_attn_weights = F.softmax(word_attn_scores, dim=-1).unsqueeze(1)
        
        sent_vectors = torch.bmm(word_attn_weights, word_hiddens).squeeze(1) # (B*S, H*2)
        sent_vectors = sent_vectors.reshape(B, S, -1) # (B, S, H*2)

        # 4. Sentence RNN
        pos_indices = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        sent_vectors = sent_vectors + self.sent_pos_emb(pos_indices)
        
        sent_hiddens, _ = self.sent_rnn(sent_vectors) 

        return word_hiddens, sent_hiddens, word_last_hidden

# --- DECODER ---

class Decoder(nn.Module):
    def __init__(self, config, vocab: Hierachy_Vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        
        self.GRU = nn.GRU(
            config.hidden_size * 2, 
            config.hidden_size,
            bidirectional=False,
            num_layers=config.layer_dim, 
            batch_first=True, 
            dropout=config.dropout
        )
        self.out = nn.Linear(config.hidden_size, vocab.vocab_size)
        
        self.attention_word = WordLevelAttention(config)
        self.attention_sent = SentenceLevelAttention(config)
        
        self.selector = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.v_s = nn.Linear(config.hidden_size, 1)
        
    def forward(self, word_hiddens, sent_hiddens, decoder_initial_states, target, extra_zeros=None, enc_batch_extend_vocab=None):
        B, S_target = target.size()
        
        decoder_input = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=self.config.device)
        decoder_hidden = decoder_initial_states
        
        all_p_final = []
        all_p_gen = []
        
        for i in range(S_target):
            p_final, decoder_hidden, p_gen = self.forward_step(
                decoder_input, decoder_hidden, word_hiddens, sent_hiddens, 
                extra_zeros, enc_batch_extend_vocab
            )
            all_p_final.append(p_final)
            all_p_gen.append(p_gen)
            decoder_input = target[:, i].unsqueeze(1) 

        all_p_final = torch.cat(all_p_final, dim=1)
        all_p_gen = torch.cat(all_p_gen, dim=1) 
        
        return all_p_final, decoder_hidden, all_p_gen

    def forward_step(self, input, states, word_hiddens, sent_hiddens, extra_zeros=None, enc_batch_extend_vocab=None):
        vocab_limit = self.embedding.num_embeddings
        unk_idx = 2
        
        safe_input = input.clone()
        
        # Xử lý OOV
        mask_oov = safe_input >= vocab_limit
        if mask_oov.any():
            safe_input[mask_oov] = unk_idx
            
        # Xử lý âm (đề phòng)
        mask_neg = safe_input < 0
        if mask_neg.any():
            safe_input[mask_neg] = unk_idx

        output_prev = self.embedding(safe_input) # Dùng safe_input
        output_prev = F.relu(output_prev)
        
        B, S, _ = sent_hiddens.size()
        curr_state = states[-1]

        # 1. Attention Mechanism
        _, p_a_w = self.attention_word(word_hiddens, curr_state, B, S)
        _, p_a_s = self.attention_sent(sent_hiddens, curr_state)
        
        # rescaled_attention có shape (B, S*W) -> tức là (B, Nd)
        rescaled_attention = rescale_attention(p_a_w, p_a_s)

        # 2. Context Vector
        word_hiddens_flat = word_hiddens.reshape(B, S * word_hiddens.size(1), -1)
        context_vector = torch.bmm(rescaled_attention.unsqueeze(1), word_hiddens_flat) 
        
        # 3. GRU Step
        rnn_input = torch.cat((output_prev, context_vector), dim=-1) 
        output, hidden = self.GRU(rnn_input, states)

        # 4. P_gen Calculation
        selector_input = torch.cat((hidden[-1], 
            output_prev.squeeze(1), 
            context_vector.squeeze(1)), dim=-1
        ) 
        p_gen = torch.sigmoid(self.v_s(torch.tanh(self.selector(selector_input))))

        # 5. Vocabulary Distribution
        p_vocab = F.softmax(self.out(output), dim=-1) 
        p_vocab_weighted = p_gen.unsqueeze(1) * p_vocab 

        # 6. Pointer Mechanism (SỬA LỖI TẠI ĐÂY)
        if extra_zeros is not None:
            p_vocab_weighted = torch.cat([p_vocab_weighted, extra_zeros], dim=-1)

        if enc_batch_extend_vocab is not None:
            # SỬA LỖI: Flatten index từ (B, S, W) -> (B, S*W) để khớp với rescaled_attention
            # rescaled_attention đã là (B, Nd)
            enc_batch_extend_vocab_flat = enc_batch_extend_vocab.reshape(B, -1)
            
            p_final = p_vocab_weighted.scatter_add(
                2, 
                enc_batch_extend_vocab_flat.unsqueeze(1), # Shape (B, 1, Nd) -> Khớp 3 chiều
                (1 - p_gen).unsqueeze(1) * rescaled_attention.unsqueeze(1)
            )
        else:
            p_final = p_vocab_weighted

        return p_final, hidden, p_gen

# --- MAIN MODEL WRAPPER ---

@META_ARCHITECTURE.register()
class AbstractiveTextSummarize(nn.Module):
    def __init__(self, config, vocab: Hierachy_Vocab):
        super().__init__()
        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2 
        self.d_model = config.d_model

        self.encoder = HierarchicalFeatureRichEncoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)
        self.loss_fn = calculate_loss 

    def reshape_word_states_to_decoder(self, word_last_hidden, B, S):
        """
        Input: (2, B*S, H_enc) -> Output: (1, B, H_dec)
        H_enc = 256, H_dec = 512
        """
        hidden = word_last_hidden.view(2, B, S, -1)
        last_sent_hidden = hidden[:, :, -1, :] 
        decoder_init_state = last_sent_hidden.permute(1, 0, 2).contiguous().view(B, -1)
        return decoder_init_state.unsqueeze(0)

    def forward(self, x, pos_ids, ner_ids, tfidf_ids, labels, extra_zeros, enc_batch_extend_vocab):
        B, S, W = x.size()
        
        # 1. Encoder
        word_hiddens, sent_hiddens, word_last_hidden = self.encoder(x, pos_ids, ner_ids, tfidf_ids)

        # 2. Reshape State
        decoder_init_states = self.reshape_word_states_to_decoder(word_last_hidden, B, S)

        # 3. Decoder
        decoder_input = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=x.device)
        decoder_hidden = decoder_init_states
        
        all_p_final = []
        all_p_gen = []
        
        S_target = labels.size(1)
        for i in range(S_target):
            p_final, decoder_hidden, p_gen = self.decoder.forward_step(
                decoder_input, decoder_hidden, word_hiddens, sent_hiddens, 
                extra_zeros, enc_batch_extend_vocab
            )
            all_p_final.append(p_final)
            all_p_gen.append(p_gen)
            decoder_input = labels[:, i].unsqueeze(1) # Teacher Forcing

        all_p_final = torch.cat(all_p_final, dim=1) # (B, S, V_ext)
        all_p_gen = torch.cat(all_p_gen, dim=1)     # (B, S) -> Tensor 2D

        # 4. Calc Loss
        total_loss = 0
        for i in range(S_target):
            # SỬA LỖI TẠI ĐÂY: Truy cập all_p_gen[:, i] thay vì all_p_gen[:, i, :]
            # Ta cần unsqueeze(1) để biến (B) thành (B, 1) cho khớp với hàm loss (nếu cần)
            # Nhưng hàm calculate_loss của bạn dùng .squeeze() bên trong nên truyền (B) hay (B,1) đều được.
            # An toàn nhất là truyền (B, 1).
            
            p_gen_step = all_p_gen[:, i].unsqueeze(1) 
            
            step_loss = self.loss_fn(
                p_final=all_p_final[:, i, :], 
                p_selection=p_gen_step,            # (B, 1)
                target_idx=labels[:, i],           # (B)
                vocab_size=self.vocab.vocab_size
            )
            total_loss += step_loss
            
        return all_p_final, total_loss / S_target
    
    def predict(self, x, pos_ids, ner_ids, tfidf_ids, enc_batch_extend_vocab, extra_zeros, max_len=None):
        self.eval()
        B, S, W = x.size()
        max_len = max_len or self.MAX_LENGTH
        device = x.device
        
        with torch.no_grad():
            word_hiddens, sent_hiddens, word_last_hidden = self.encoder(x, pos_ids, ner_ids, tfidf_ids)
            states = self.reshape_word_states_to_decoder(word_last_hidden, B, S)
            
            current_input = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=device)
            decoded_indices = []
            
            for t in range(max_len):
                p_final, states, _ = self.decoder.forward_step(
                    current_input, states, word_hiddens, sent_hiddens,
                    extra_zeros, enc_batch_extend_vocab
                )
                prediction = p_final.argmax(dim=-1) 
                decoded_indices.append(prediction)
                current_input = prediction.clone()
                current_input[current_input >= self.vocab.vocab_size] = self.vocab.unk_idx
                
            decoded_indices = torch.cat(decoded_indices, dim=1)
            
        return decoded_indices
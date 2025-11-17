import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE
from collections import namedtuple
class Encoder(nn.Module):
    
    def __init__(self, config, vocab, shared_embedding):
        super().__init__()
        self.embedding = shared_embedding
        self.vocab = vocab
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            bidirectional=True,
            num_layers=config.encoder.num_layers,
            batch_first=True,
            device=config.device
        )
        self.linear = nn.Linear(config.hidden_size*2, 
            config.hidden_size, 
            device=config.device    
        )

    def forward(self, input):
        # 1. 'encoder_input' là tensor GỐC, chứa OOV indices
        encoder_input = input 
        
        # 2. Tạo 'embedding_input', một bản sao an toàn cho lớp embedding
        embedding_input = input.clone()
        embedding_input[embedding_input >= self.vocab.vocab_size] = self.vocab.unk_idx
        
        # 3. Đưa bản an toàn vào embedding
        embedded = self.embedding(embedding_input) 
        encoder_output, (h_n, c_n) = self.lstm(embedded)
        
        # encoder_output: (B, seq_len, H*2)
        # h_n, c_n: (num_layers * 2, B, H) -> (4, B, H)

        # Mục tiêu: Tạo state (h_n, c_n) với shape (1, B, H) 
        # cho Decoder (vốn có num_layers=1)
        
        # Lấy hidden state của layer CUỐI CÙNG (layer 2)
        # Forward: index -2 (Layer 2 Fwd)
        # Backward: index -1 (Layer 2 Bwd)
        h_n_fwd = h_n[-2, :, :] # Shape: (B, H)
        h_n_bwd = h_n[-1, :, :] # Shape: (B, H)
        
        # Nối 2_directions lại: (B, H*2)
        h_n_combined = torch.cat((h_n_fwd, h_n_bwd), dim=1)
        
        # Lấy cell state của layer CUỐI CÙNG (tương tự h_n)
        c_n_fwd = c_n[-2, :, :] # Shape: (B, H)
        c_n_bwd = c_n[-1, :, :] # Shape: (B, H)
        
        # Nối 2_directions lại: (B, H*2)
        c_n_combined = torch.cat((c_n_fwd, c_n_bwd), dim=1)

        # Decoder của bạn (theo lỗi) chỉ có 1 layer,
        # nên chúng ta cần thêm 1 chiều ở đầu (num_layers=1)
        h_n_final = h_n_combined.unsqueeze(0) # Shape: (1, B, H*2)
        c_n_final = c_n_combined.unsqueeze(0) # Shape: (1, B, H*2)
        
        states = (h_n_final, c_n_final) # <--- Shape (1, B, H)

        max_src_index = input.max().item()
        num_oov_in_batch = max(0, max_src_index - self.vocab.vocab_size + 1)

        return encoder_output, states, encoder_input, num_oov_in_batch

class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab, shared_embedding):
        super().__init__()
        self.embedding = shared_embedding
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size*2,
            num_layers=config.decoder.num_layers,
            batch_first=True,
            dropout=0.5,
            device=config.device,
            bidirectional=False # <<< THAY ĐỔI 1: Chuyển sang False
        )
        self.reduce_state = nn.Linear(
            config.hidden_size*4, # <<< THAY ĐỔI 2: Input giảm từ H*8 xuống H*4
            config.hidden_size*2,
            device=config.device
        )
        self.attn = BahdanauAttention(config) # Giả định lớp này tồn tại
        self.linear1 = nn.Linear(config.hidden_size * 4, 
            config.hidden_size * 2, 
            device=config.device,
            bias = True
        )
        self.out = nn.Linear(config.hidden_size * 2, 
            vocab.vocab_size, 
            device=config.device,
            bias = True
        ) 

        # Pointer-Generator parameters
        self.linear_context = nn.Linear(
            config.hidden_size*2,
            config.hidden_size,
            device=config.device,
        )
        self.linear_decoder_state = nn.Linear(
            config.hidden_size*2,
            config.hidden_size,
            device=config.device,
        )
        self.linear_decoder_input = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            device=config.device,
        )
        self.p_gen_linear = nn.Linear(
            config.hidden_size,
            1,
            device=config.device,
        )
        self.vocab = vocab
        self.sigmoid = nn.Sigmoid()
        # self.prj_hidden và self.prj_memory không được dùng trong forward, bỏ qua
        
        # Thêm self.b_ptr (thiếu trong code gốc)
        self.b_ptr = nn.Parameter(torch.zeros(1, device=config.device))


    def forward(self, input, states, target_tensor=None, encoder_outputs=None, num_oov_in_batch=0, encoder_input=None):
        # Initial states
        decoder_hidden, decoder_memory = states # Shape (num_layers, B, H*2)
        
        decoder_outputs = []
        target_len = target_tensor.shape[-1]
        decoder_input = input  # (batch_size, 1)
        coverage = torch.zeros(encoder_outputs.size(0), encoder_outputs.size(1), device=encoder_outputs.device)  # (B, S)
        attention_weights_list = []
        coverages_list = []

        for i in range(target_len):
            decoder_output, (decoder_hidden, decoder_memory), attention_weights, coverage = self.forward_step(
                decoder_input, 
                (decoder_hidden, decoder_memory),
                encoder_outputs=encoder_outputs,
                encoder_input=encoder_input,
                coverage=coverage,
                num_oov_in_batch=num_oov_in_batch
            )
            decoder_outputs.append(decoder_output)
            attention_weights_list.append(attention_weights)
            coverages_list.append(coverage)
            # Teacher forcing: Feed the target as the next input
            decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing

        decoder_outputs = torch.stack(decoder_outputs, dim=1) # (batch_size, target_len, extended_vocab_size)
        attention_weights_list = torch.stack(attention_weights_list, dim=1) # Shape (B, T, S)
        coverages_list = torch.stack(coverages_list, dim=1) # Shape (B, T, S)
        return decoder_outputs, (decoder_hidden, decoder_memory), attention_weights_list, coverages_list


    def forward_step(self, input, states, encoder_outputs, num_oov_in_batch=0, encoder_input=None, coverage=None):
        embedding_input = input.clone()
        embedding_input[embedding_input >= self.vocab.vocab_size] = self.vocab.unk_idx
        
        # 2. Đưa bản an toàn vào embedding
        embedded = self.embedding(embedding_input) # <--- ĐÃ AN TOÀN
        # ----------------------------------------------------
        
        # embedded: (batch_size, 1, hidden_size)

        output, (h_n, c_n) = self.lstm(embedded, states)
        # output : (batch_size, 1, hidden_size*2)        # <<< THAY ĐỔI 3 (Comment)
        # h_n : (num_layers, batch_size, hidden_size*2) # <<< THAY ĐỔI 4 (Comment)
        # c_n : (num_layers, batch_size, hidden_size*2) # <<< THAY ĐỔI 5 (Comment)

        # Lấy state của lớp cuối cùng (không cần gộp fwd/bwd)
        h_last = h_n[-1] # (B, H*2)                      # <<< THAY ĐỔI 6
        c_last = c_n[-1] # (B, H*2)                      # <<< THAY ĐỔI 7

        s_t = h_last  # Sử dụng h_last làm s_t cho attention

        C_t, A_ti, coverage = self.attn(s_t = s_t, h_i = encoder_outputs, coverage=coverage) 
        # C_t: (B, H*2)
        # A_ti: (B, S)

        # Pointer-Generator
        p_gen = self.sigmoid(
                    self.p_gen_linear(
                        self.linear_context(C_t) + # (batch_size, hidden_size)
                        self.linear_decoder_state(s_t) + # (batch_size, hidden_size)
                        self.linear_decoder_input(embedded.squeeze(1)) + # (batch_size, hidden_size)
                        self.b_ptr # (1, )
                    )
                )
        # p_gen: (batch_size, 1)
        
        #P_vocab
        concat_input = torch.cat((s_t, C_t), dim=-1)  # (B, hidden_size*4)
        hidden = F.relu(self.linear1(concat_input)) # (B, hidden_size*2)
        logits = self.out(hidden) # (B, vocab_size)
        Pvocab = F.softmax(logits, dim=-1) # Pvocab: (B, vocab_size)
        
        # Copy distribution
        extended_vocab_size = self.vocab.vocab_size + num_oov_in_batch
        batch_size = encoder_input.size(0)
        
        extended_P_vocab = torch.zeros(batch_size, extended_vocab_size, device=encoder_input.device)
        extended_P_vocab[:, :self.vocab.vocab_size] = Pvocab 
        copy_dist = torch.zeros_like(extended_P_vocab)  
        copy_dist = copy_dist.scatter_add(dim=1, index=encoder_input, src=A_ti)

        # Final P
        final_dist = p_gen * extended_P_vocab + (1 - p_gen) * copy_dist  # (B, extended_vocab_size)
        return final_dist, (h_n, c_n), A_ti, coverage

class DecoderClosedBook(nn.Module):
    def __init__(self, config, vocab: Vocab, shared_embedding):
        """
        Khởi tạo Closed-Book Decoder.
        
        Args:
            config: Đối tượng cấu hình (chứa hidden_size, num_layers, device...)
            vocab: Đối tượng Vocab (chứa vocab_size, bos_idx)
        """
        super().__init__()
        self.vocab = vocab
        self.embedding = shared_embedding   
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size * 2,
            # QUAN TRỌNG: Đặt num_layers=1 để khớp với state (1, B, H*2)
            # mà Encoder.forward của bạn trả về.
            num_layers=config.decoder.num_layers, 
            batch_first=True,
            device=config.device,
            bidirectional=False # Yêu cầu chính: Unidirectional
        )

        # 3. Lớp output (ánh xạ từ hidden_size*2 ra vocab_size)
        self.out = nn.Linear(
            config.hidden_size * 2, 
            vocab.vocab_size, 
            device=config.device,
            bias=True
        )

    def forward(self, states, target_tensor):
        """
        Chạy vòng lặp decoder (luôn dùng teacher forcing vì đây là lúc training).

        Input:
            states: (h_0, c_0) - Trạng thái ban đầu từ encoder.
                    h_0: (1, batch_size, hidden_size*2)
                    c_0: (1, batch_size, hidden_size*2)
            target_tensor: (batch_size, target_len) - Các từ target
        
        Output:
            all_logits: (batch_size, target_len, vocab_size)
        """
        decoder_hidden, decoder_memory = states
        target_len = target_tensor.shape[1]
        batch_size = target_tensor.shape[0]

        # Khởi tạo token [SOS] làm input đầu tiên
        decoder_input = torch.empty(
            batch_size, 1,
            dtype=torch.long,
            device=target_tensor.device
        ).fill_(self.vocab.bos_idx)
        
        all_logits = []

        # Lặp qua từng bước thời gian của chuỗi target
        for i in range(target_len):
            # Chạy một bước giải mã
            logits_step, (decoder_hidden, decoder_memory) = self.forward_step(
                decoder_input,
                (decoder_hidden, decoder_memory)
            )
            
            # Lưu trữ logits (chưa qua softmax)
            all_logits.append(logits_step)
            
            # Teacher forcing:
            # Lấy từ target đúng làm đầu vào cho bước tiếp theo
            decoder_input = target_tensor[:, i].unsqueeze(1) 

        # Ghép tất cả logits lại
        # Shape: (batch_size, target_len, vocab_size)
        all_logits = torch.stack(all_logits, dim=1) 
        
        return all_logits

    def forward_step(self, input, states):
        """
        Thực hiện một bước giải mã.
        Input:
            input (B, 1)
            states: (h_prev, c_prev) - Shape (1, B, H*2)
        Output:
            logits: (B, vocab_size)
            (h_n, c_n): Trạng thái mới - Shape (1, B, H*2)
        """
        embedding_input = input.clone()
        embedding_input[embedding_input >= self.vocab.vocab_size] = self.vocab.unk_idx
        
        # 2. Đưa bản an toàn vào embedding
        embedded = self.embedding(embedding_input) # <--- ĐÃ AN TOÀN

        # 2. Đưa qua LSTM
        # output: (B, 1, H*2)
        # (h_n, c_n): (1, B, H*2)
        output, (h_n, c_n) = self.lstm(embedded, states)
        
        # 3. Tính Logits
        # Lấy output của LSTM (B, 1, H*2), bỏ chiều seq_len=1
        # và đưa qua lớp Linear để ra vocab_size
        # output.squeeze(1): (B, H*2)
        logits = self.out(output.squeeze(1)) 
        # logits: (B, vocab_size)
        
        return logits, (h_n, c_n)

class BahdanauAttention(nn.Module): # Attention Bahdanau-style
    def __init__(self, config):
        super().__init__()
        self.W_h = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.W_s = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.W_c = nn.Linear(1, config.hidden_size, bias=False)
        self.v_a = nn.Linear(config.hidden_size, 1, bias=True)

    def forward(self, s_t, h_i, coverage): 
        """
        Inputs:
            s_t: (B, H*2)
            h_i: (B, S, H*2)
            coverage: (B, S)
        Outputs:
            context_vector: (B, H*2)
            attention_weights: (B, S)
            coverage: (B, S)
        """
        Wi_hi = self.W_h(h_i)  # (B, S, hidden_size)
        Ws_s = self.W_s(s_t)  # (B, hidden_size)
        coverage_features = self.W_c(coverage.unsqueeze(2)) # (B, S, 1) -> (B, S, hidden_size)
        E_ti = self.v_a(torch.tanh(Wi_hi + Ws_s.unsqueeze(1) + coverage_features)).squeeze(-1)  # (B, S)
        A_ti = F.softmax(E_ti, dim=-1) # (B, S)
        C_t = torch.bmm(A_ti.unsqueeze(1), h_i).squeeze(1)  # (B, 1, S) * (B, S, H) -> (B, 1, H) -> (B, H*2)

        coverage = coverage + A_ti  # (B, S)

        return C_t, A_ti, coverage  # context_vector (B, H*2), attention_weights (B, S), coverage (B, S)

class LossFunc(nn.Module):
    def __init__(self, vocab_size, lambda_cov=1.0, pad_idx=0):
        super().__init__()
        self.lambda_cov = lambda_cov
        self.vocab_size = vocab_size
        self.NLloss = nn.NLLLoss(ignore_index=pad_idx, reduction='mean') # Sửa: dùng 'mean'
        self.pad_idx = pad_idx # Lưu pad_idx

    def forward(self, final_dists, target_tensor, attention_dists, coverages):
        log_probs = torch.log(final_dists + 1e-12)
        log_probs_flat = log_probs.view(-1, log_probs.size(-1))
        target_flat = target_tensor.view(-1)
        
        nll_loss = self.NLloss(log_probs_flat, target_flat)
        loss = nll_loss

        if self.lambda_cov > 0:
            attention_dists_stacked = attention_dists # (B, T, S)
            coverages_stacked = coverages             # (B, T, S)
            
            # --- (SỬA LỖI LOGIC 3: COVERAGE LOSS) ---
            # c_{t-1} = c_t - a_t
            # (Giả sử c_{-1} là 0)
            # Tạo c_{t-1} bằng cách dịch chuyển c_t
            B, T, S = coverages_stacked.size()
            # c_0 = 0 (B, 1, S)
            c_prev_0 = torch.zeros(B, 1, S, device=coverages_stacked.device)
            # c_{t-1} (B, T-1, S)
            c_prev_t_minus_1 = coverages_stacked[:, :-1, :]
            # Ghép lại -> (B, T, S)
            coverages_prev = torch.cat([c_prev_0, c_prev_t_minus_1], dim=1)

            # Tính min(a_t, c_{t-1})
            cov_loss_all_steps = torch.min(attention_dists_stacked, coverages_prev)
            # 1. Tạo mask padding
            # target_flat có shape (B*T,)
            # Chúng ta muốn mask cho cov_loss_all_steps (B, T, S)
            # -> mask phải có shape (B, T)
            padding_mask = (target_tensor != self.pad_idx) # Shape (B, T)

            # 2. Sum loss trên chiều source (S)
            cov_loss_per_step = torch.sum(cov_loss_all_steps, dim=2) # Shape (B, T)

            # 3. Áp dụng padding mask
            cov_loss_per_step_masked = cov_loss_per_step * padding_mask.float()

            # 4. Tính loss trung bình (chính xác)
            # Sum toàn bộ loss và chia cho số token không phải pad
            num_non_pad_tokens = padding_mask.sum()
            if num_non_pad_tokens > 0:
                cov_loss = torch.sum(cov_loss_per_step_masked) / num_non_pad_tokens
            else:
                cov_loss = torch.tensor(0.0, device=final_dists.device)
            
            loss += self.lambda_cov * cov_loss
            # --- (HẾT PHẦN SỬA LẠI) ---

        # Trả về loss (total), nll_loss, và coverage_loss
        return loss, nll_loss, (loss - nll_loss) # (cov_loss bây giờ được tính trong 'loss')
    

BeamHypothesis = namedtuple("BeamHypothesis", 
    ["tokens", "score", "state", "coverage"])


@META_ARCHITECTURE.register()
class ClosedBookModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab_size = vocab.vocab_size
        self.shared_embedding = nn.Embedding(
            vocab.vocab_size, config.hidden_size, device=config.device
        )
        self.encoder = Encoder(config, vocab, self.shared_embedding)
        self.attn_decoder = Decoder(config, vocab, self.shared_embedding)
        self.cb_decoder = DecoderClosedBook(config, vocab, self.shared_embedding)

        self.gamma = config.gamma
        self.d_model = config.d_model
        self.device = config.device 
        self.config = config
        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2  # + 2 for bos and eos tokens
        self.pgn_loss = LossFunc(vocab_size=vocab.vocab_size, pad_idx=vocab.pad_idx, lambda_cov=config.lambda_coverage)
        self.cb_loss_func = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    def forward(self, src, labels):
        encoder_outputs, hidden_states, encoder_input, num_oov_in_batch = self.encoder(src)
        
        batch_size = src.size(0)
        decoder_input = torch.empty(
            batch_size, 1,
            dtype=torch.long,
            device=src.device
        ).fill_(self.vocab.bos_idx)
        
        pgn_outputs, _, attn_list, cov_list = self.attn_decoder(
            decoder_input,
            hidden_states,
            labels, 
            encoder_outputs=encoder_outputs,
            num_oov_in_batch=num_oov_in_batch,
            encoder_input=encoder_input
        )

        # --- (THAY ĐỔI 1): Lấy cả 3 giá trị loss ---
        loss_pgn, nll_loss, cov_loss = self.pgn_loss(
            final_dists = pgn_outputs,
            target_tensor = labels, 
            attention_dists = attn_list,
            coverages = cov_list
        )

        cb_outputs = self.cb_decoder(
            hidden_states,
            labels 
        ) # Shape: (B, T, VocabSize)

        cb_labels = labels.clone() 
        cb_labels[cb_labels >= self.vocab_size] = self.vocab.unk_idx 
        
        loss_cb = self.cb_loss_func(
            cb_outputs.view(-1, self.vocab_size),
            cb_labels.view(-1)
        )
        
        total_loss = (1 - self.gamma) * loss_pgn + self.gamma * loss_cb
        
        # --- (THAY ĐỔI 2): Trả về 4 giá trị ---
        return pgn_outputs, total_loss
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện dự đoán bằng Greedy Search (chỉ hỗ trợ batch_size = 1).
        """
        # 1. Chạy Encoder (Giống hệt beam search)
        # Giả sử x có batch_size = 1
        encoder_outputs, hidden_states, encoder_input, num_oov_in_batch = self.encoder(x)
        
        # 2. Khởi tạo
        # Trạng thái ban đầu
        current_state = hidden_states
        current_coverage = torch.zeros(encoder_outputs.size(0), encoder_outputs.size(1), device=encoder_outputs.device) # (1, S)
        
        # Token [BOS] làm input đầu tiên
        decoder_input = torch.tensor([[self.vocab.bos_idx]], device=self.device, dtype=torch.long) # Shape (1, 1)

        output_tokens = []

        # 3. Chạy vòng lặp giải mã
        for _ in range(self.MAX_LENGTH):
            # 3.1. Chạy 1 bước PGN Decoder
            final_dist, new_state, _, new_coverage = self.attn_decoder.forward_step(
                decoder_input,
                current_state,
                encoder_outputs=encoder_outputs,
                num_oov_in_batch=num_oov_in_batch,
                encoder_input=encoder_input,
                coverage=current_coverage
            ) # final_dist shape (1, extended_vocab_size)
            
            # 3.2. Lấy token tốt nhất (Greedy)
            # Thay vì topk, chúng ta dùng argmax
            _, next_token = torch.max(final_dist, dim=1) # Shape (1)
            next_token_item = next_token.item()
            
            # 3.3. Lưu token và kiểm tra [EOS]
            output_tokens.append(next_token_item)
            if next_token_item == self.vocab.eos_idx:
                break

            # 3.4. Chuẩn bị cho vòng lặp tiếp theo
            current_state = new_state
            current_coverage = new_coverage
            
            # Input cho bước tiếp theo chính là token vừa sinh ra
            decoder_input = next_token.unsqueeze(0) # Shape (1, 1)

        # 4. Hoàn tất
        # Trả về Tensor, unsqueeze(0) để tạo batch_size=1
        return torch.tensor(output_tokens, device=self.device).unsqueeze(0)

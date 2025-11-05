import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class Encoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab.vocab_size, config.hidden_size, device=config.device
        )
        self.vocab = vocab
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
            device=config.device
        )
        self.linear = nn.Linear(config.hidden_size*2, 
            config.hidden_size, 
            device=config.device    
        )

    def forward(self, input):
        """
        Input: (batch_size, seq_len)
        Output: 
            output: (batch_size, seq_len, hidden_size * 2)
            h_n: (2*num_layers, hidden_size)
            c_n: (2*num_layers, hidden_size)
        """
        embedded = self.embedding(input) 
        encoder_output, (h_n, c_n) = self.lstm(embedded)
        encoder_output = self.linear(encoder_output)
        encoder_input = input
        # output: (batch_size, seq_len, hidden_size)
        # h_n: (2*num_layers, batch_size, hidden_size)
        # c_n: (2*num_layers, batch_size, hidden_size)
        max_src_index = input.max().item()
        num_oov_in_batch = max(0, max_src_index - self.vocab.vocab_size + 1)
        return encoder_output, (h_n, c_n), encoder_input, num_oov_in_batch


class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab.vocab_size,
            config.hidden_size,
            device=config.device
        )
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.7,
            device=config.device,
            bidirectional=False
        )
        self.attn = BahdanauAttention(config)
        self.linear1 = nn.Linear(config.hidden_size * 2, 
            config.hidden_size, 
            device=config.device,
            bias = True
        )
        self.out = nn.Linear(config.hidden_size, 
            vocab.vocab_size, 
            device=config.device,
            bias = True
        ) 

        # Pointer-Generator parameters
        self.linear_context = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            device=config.device,
        )
        self.linear_decoder_state = nn.Linear(
            config.hidden_size,
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
        self.b_ptr = nn.Parameter(torch.zeros(1, device=config.device))
        self.sigmoid = nn.Sigmoid()
        self.prj_hidden = nn.Linear(config.hidden_size*2, config.hidden_size, device=config.device)
        self.prj_memory = nn.Linear(config.hidden_size*2, config.hidden_size, device=config.device)
    def forward(self, input, states, target_tensor=None, encoder_outputs=None, num_oov_in_batch=0, encoder_input=None):
        """
        Input: 
            input: (batch_size, 1)
            states: (h_0, c_0)
                h_0: (2, batch_size, hidden_size)
                c_0: (2, batch_size, hidden_size)
            target_tensor: (batch_size, target_len)
            encoder_outputs: (batch_size, seq_len, hidden_size)
            num_oov_in_batch: (int)
        Output:
            output: (batch_size, target_len, vocab_size)
            h_n: (num_layers, batch_size, hidden_size)
            c_n: (num_layers, batch_size, hidden_size)
        """
        #Initial states
        decoder_hidden, decoder_memory = states
        decoder_hidden = torch.cat((decoder_hidden[0], decoder_hidden[1]), dim=-1) # (batch_size, hidden_size*2)
        decoder_memory = torch.cat((decoder_memory[0], decoder_memory[1]), dim=-1) # (batch_size, hidden_size*2)
        decoder_hidden = self.prj_hidden(decoder_hidden).unsqueeze(0)  # (1, batch_size, hidden_size)
        decoder_memory = self.prj_memory(decoder_memory).unsqueeze(0)  # (1, batch_size, hidden_size)

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
        return decoder_outputs, (decoder_hidden, decoder_memory), attention_weights_list, coverages_list

    def forward_step(self, input, states, encoder_outputs, num_oov_in_batch=0, encoder_input=None, coverage=None):
        """
        Input: 
            input (B, 1)
            states: (h_0, c_0)
                h_0: (1, batch_size, hidden_size)
                c_0: (1, batch_size, hidden_size)
            encoder_outputs: (B, S, hidden_size)
            coverage: (B, S)
        Output:
        """

        embedded = self.embedding(input) # (batch_size, 1, hidden_size)
        output, (h_n, c_n) = self.lstm(embedded, states)
        # output : (batch_size, 1, hidden_size)
        # h_n : (1, batch_size, hidden_size)
        # c_n : (1, batch_size, hidden_size)
        
        # Attention
        if coverage is None:
            coverage = torch.zeros(encoder_outputs.size(0), encoder_outputs.size(1), device=encoder_outputs.device)  # (B, S)
        decoder_state = h_n[-1] # (batch_size, hidden_size)
        context_vector, attention_weights, coverage = self.attn(decoder_state, encoder_outputs, coverage) # decoder_state: (batch_size, hidden_size)
        # context_vector: (batch_size, hidden_size), attention_weights: (batch_size, seq_len)


        # Pointer-Generator
        p_gen = self.sigmoid(
                    self.p_gen_linear(
                        self.linear_context(context_vector) + # (batch_size, hidden_size)
                        self.linear_decoder_state(decoder_state) + # (batch_size, hidden_size)
                        self.linear_decoder_input(embedded.squeeze(1)) + # (batch_size, hidden_size)
                        self.b_ptr # (1, )
                    )
                )
        # p_gen: (batch_size, 1)
        
        #P_vocab
        concat_input = torch.cat((decoder_state, context_vector), dim=-1)  # (batch_size, hidden_size*2)
        hidden = F.relu(self.linear1(concat_input))
        logits = self.out(hidden)
        Pvocab = F.softmax(logits, dim=-1) # Pvocab: (B, vocab_size)
        
        # Copy distribution
        extended_vocab_size = self.vocab.vocab_size + num_oov_in_batch
        batch_size = encoder_input.size(0)
        
        extended_P_vocab = torch.zeros(batch_size, extended_vocab_size, device=encoder_input.device)
        # extended_P_vocab: (B, extended_vocab_size)
        extended_P_vocab[:, :self.vocab.vocab_size] = Pvocab # Sao chép Pvocab vào phần từ vựng gốc [bắt đầu : kết thúc]
        copy_dist = torch.zeros_like(extended_P_vocab)  # (B, extended_vocab_size)
        copy_dist = copy_dist.scatter_add(1, encoder_input, attention_weights)

        # Final P
        final_dist = p_gen * extended_P_vocab + (1 - p_gen) * copy_dist  # (B, extended_vocab_size)
        return final_dist, (h_n, c_n), attention_weights, coverage

class BahdanauAttention(nn.Module): # Attention Bahdanau-style
    def __init__(self, config):
        super().__init__()
        self.W_h = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_s = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_c = nn.Linear(1, config.hidden_size)
        self.b_attn = nn.Parameter(torch.zeros(1, config.hidden_size)) # (1, hidden_size)
        self.v_a = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, prev_decoder_hidden, encoder_outputs, coverage): 
        """
        Inputs:
            prev_decoder_hidden: (batch_size, hidden_size)
            encoder_outputs: (B, S, hidden_size)
            coverage: (B, S)
        Outputs:
            context_vector: (B, hidden_size)
            attention_weights: (B, S)
            coverage: (B, S)
        """
        Wi_hi = self.W_h(encoder_outputs)  # (B, S, hidden_size)
        Ws_prev_s = self.W_s(prev_decoder_hidden)  # (B, hidden_size)
        Wc_ci = self.W_c(coverage.unsqueeze(-1))  # (B, S, hidden_size)


        E_ti = self.v_a(torch.tanh(Wi_hi + Ws_prev_s.unsqueeze(1) + Wc_ci)).squeeze(-1)  # (B, S)
        A_ti = F.softmax(E_ti, dim=-1) # (B, S)
        C_t = torch.bmm(A_ti.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, 1, S) * (B, S, H) -> (B, 1, H) -> (B, H)

        coverage = coverage + A_ti  # (B, S)

        return C_t, A_ti, coverage  # context_vector (B, H), attention_weights (B, S), coverage (B, S)

class LossFunc(nn.Module):
    def __init__(self, vocab_size, lambda_cov=1.0):
        super().__init__()
        self.lambda_cov = lambda_cov
        self.vocab_size = vocab_size
        self.loss = nn.NLLLoss(ignore_index=0, reduction='mean')  # giả sử 0 là padding

    def forward(self, final_dists, target_tensor, attention_dists, coverages):
        """
        Inputs:
            final_dists: list hoặc tensor (B, extended_vocab_size)
            target_tensor: (B, T)
            attention_dists: list (T phần tử, mỗi phần tử (B, S))
            coverages: list (T phần tử, mỗi phần tử (B, S))
        """
        eps = 1e-12
        batch_size, seq_len = target_tensor.size()

        nll_loss_total = 0.0
        cov_loss_total = 0.0

        for t in range(seq_len):
            final_dist_t = final_dists[:, t, :]  # (B, V)
            attn_t = attention_dists[t] if isinstance(attention_dists, list) else attention_dists
            cov_t = coverages[t] if isinstance(coverages, list) else coverages

            log_probs = torch.log(final_dist_t + eps)  # (B, V)
            nll_t = self.loss(log_probs, target_tensor[:, t])  # (B,)
            nll_loss_total += nll_t

            # Coverage loss: ∑ min(a_ti, c_ti)
            if attn_t is not None and cov_t is not None:
                cov_t_loss = torch.mean(torch.sum(torch.min(attn_t, cov_t), dim=1))
                cov_loss_total += cov_t_loss

        nll_loss_total /= seq_len
        cov_loss_total /= seq_len

        total_loss = nll_loss_total + self.lambda_cov * cov_loss_total

        return total_loss, nll_loss_total, cov_loss_total

@META_ARCHITECTURE.register()
class ClosedBookModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab_size = vocab.vocab_size
        
        self.encoder = Encoder(config, vocab)
        self.decoder = Decoder(config, vocab)

        self.d_model = config.d_model
        self.device = config.device 
        self.config = config
        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2  # + 2 for bos and eos tokens
        self.loss = LossFunc(vocab_size=vocab.vocab_size, lambda_cov=config.lambda_coverage)

    def forward(self, src, labels):
        """
        Forward pass for training
        
        Args:
            src (torch.Tensor): Source sequences of shape (batch_size, src_len)
            labels (torch.Tensor): Target sequences of shape (batch_size, tgt_len)
            
        Returns:
            tuple:
                - decoder_outputs (torch.Tensor): Output logits of shape (batch_size, tgt_len, vocab_size)
                - loss (torch.Tensor): Scalar loss value
        """
        # Encode source sequence
        encoder_outputs, hidden_states, encoder_input, num_oov_in_batch = self.encoder(src)
        
        # Initialize decoder input
        batch_size = src.size(0)
        decoder_input = torch.empty(
            batch_size, 1,
            dtype=torch.long,
            device=src.device
        ).fill_(self.vocab.bos_idx)
        
        # Decode with teacher forcing
        decoder_outputs, _, attention_weights_list, coverage_list  = self.decoder(
            decoder_input,
            hidden_states,
            labels,
            encoder_outputs=encoder_outputs,
            num_oov_in_batch=num_oov_in_batch,
            encoder_input=encoder_input
        )

        # Calculate loss
        loss, _, _ = self.loss(
            final_dists = decoder_outputs,
            target_tensor = labels,
            attention_dists = attention_weights_list,
            coverages = coverage_list
        )
        
        return decoder_outputs, loss
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs, hidden_states, encoder_input, num_oov_in_batch = self.encoder(x)
        batch_size = encoder_outputs.size(0)
        
        decoder_input = torch.empty(
            batch_size, 1,
            dtype=torch.long,
            device=x.device
        ).fill_(self.vocab.bos_idx)
        outputs = []
        coverage = None
        h_n, c_n = hidden_states 
        
        # 1. Ghép 2 chiều (forward/backward) của BiLSTM
        h_n_concat = torch.cat((h_n[0], h_n[1]), dim=-1) # (B, H*2)
        c_n_concat = torch.cat((c_n[0], c_n[1]), dim=-1) # (B, H*2)
         
        # 2. Dùng linear layer của DECODER để chiếu về đúng shape (1, B, H)
        decoder_h = self.decoder.prj_hidden(h_n_concat).unsqueeze(0) # (1, B, H)
        decoder_c = self.decoder.prj_memory(c_n_concat).unsqueeze(0) # (1, B, H)
        decoder_states = (decoder_h, decoder_c)
        for _ in range(self.MAX_LENGTH):
            decoder_output, decoder_states, _, coverage = self.decoder.forward_step(
                decoder_input,
                decoder_states,
                encoder_outputs=encoder_outputs,
                num_oov_in_batch=num_oov_in_batch,
                encoder_input=encoder_input,
                coverage=coverage
            )
            
            top_idx = decoder_output.argmax(dim=-1) # Shape (B,)
            # 2. Lưu token này vào danh sách output
            # (Phải unsqueeze để (B,) -> (B, 1) rồi mới cat ở cuối)
            outputs.append(top_idx.unsqueeze(1)) 

            # 3. Chuẩn bị input cho bước tiếp theo
            decoder_input = top_idx.clone()
            # Map OOV (>= vocab_size) về UNK (để không crash embedding)
            decoder_input[decoder_input >= self.vocab.vocab_size] = self.vocab.unk_idx
            decoder_input = decoder_input.unsqueeze(1) # Shape (B, 1)

            if (top_idx == self.vocab.eos_idx).all():
                break
        outputs = torch.cat(outputs, dim=1)

        return outputs

import torch
import torch.nn as nn
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab

class Encoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        
        self.lstm = nn.LSTM(config.hidden_size, 
                            config.hidden_size, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True)

        self.W_h = nn.Linear(config.hidden_size * 2, config.hidden_size * 2, bias=False)
        self.hidden_dim = config.hidden_size

    def forward(self, input, seq_lens):
        embedded = self.embedding(input)
        encoder_outputs, hidden = self.lstm(embedded)
        
        encoder_feature = encoder_outputs.reshape(-1, 2*self.hidden_dim)
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden

class ReduceState(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(ReduceState, self).__init__()
        self.reduce_h = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.reduce_c = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_dim = config.hidden_size

    def forward(self, hidden):
        h, c = hidden
        h_in = h.transpose(0, 1).reshape(-1, self.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).reshape(-1, self.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))

class Attention(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(Attention, self).__init__()
        self.decode_proj = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        self.v = nn.Linear(config.hidden_size * 2, 1, bias=False)
        self.is_coverage = config.is_coverage
        if self.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_size * 2, bias=False)
        self.hidden_dim = config.hidden_size

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat)
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()
        dec_fea_expanded = dec_fea_expanded.reshape(-1, n)

        att_features = encoder_feature + dec_fea_expanded
        if self.is_coverage:
            coverage_input = coverage.view(-1, 1)
            coverage_feature = self.W_c(coverage_input)
            att_features = att_features + coverage_feature

        e = F.tanh(att_features)
        scores = self.v(e)
        scores = scores.reshape(-1, t_k)

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)
        c_t = torch.bmm(attn_dist, encoder_outputs)
        c_t = c_t.reshape(-1, self.hidden_dim * 2)

        attn_dist = attn_dist.reshape(-1, t_k)

        if self.is_coverage:
            coverage = coverage.reshape(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(Decoder, self).__init__()
        self.attention_network = Attention(config, vocab)
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.x_context = nn.Linear(config.hidden_size * 2 + config.hidden_size, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.pointer_gen = config.pointer_gen

        if self.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_size * 4 + config.hidden_size, 1)

        self.out1 = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.out2 = nn.Linear(config.hidden_size, vocab.vocab_size)
        self.hidden_dim = config.hidden_size

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):
        
        # ⭐ Inference warm-up
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim),
                                 c_decoder.view(-1, self.hidden_dim)), 1)
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                          enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.dropout(self.embedding(y_t_1))
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.reshape(-1, self.hidden_dim),
                             c_decoder.reshape(-1, self.hidden_dim)), 1)
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature, 
                                                              enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if self.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.reshape(-1, self.hidden_dim), c_t), 1)
        output = self.out1(output)
        output = F.relu(output)
        output = self.out2(output)
        vocab_dist = F.softmax(output, dim=1)

        if self.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class ClosedbookDecoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(ClosedbookDecoder, self).__init__()
        
        # ⭐ Core components theo paper
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.x_context = nn.Linear(config.hidden_size * 2 + config.hidden_size, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.max_src_len = vocab.max_sentence_length
        # ⭐ Internal memory - key component từ paper
        self.internal_memory = nn.Parameter(
            torch.randn(self.max_src_len, config.hidden_size * 2)
        )
        
        # ⭐ Memory attention mechanism
        self.memory_attention = Attention(config, vocab)
        
        # ⭐ Pointer-generator setup
        self.pointer_gen = config.pointer_gen
        if self.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_size * 4 + config.hidden_size, 1)
        
        # ⭐ Output layers
        self.out1 = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.out2 = nn.Linear(config.hidden_size, vocab.vocab_size)
        
        # ⭐ Coverage setup
        self.is_coverage = config.is_coverage
        if self.is_coverage:
            self.coverage_layer = nn.Linear(1, config.hidden_size * 2, bias=False)
        
        self.hidden_dim = config.hidden_size

    def forward(self, y_t_1, s_t_1, c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):
        batch_size = y_t_1.size(0)
        
        # ⭐ Expand internal memory cho batch
        encoder_outputs = self.internal_memory.unsqueeze(0).expand(batch_size, -1, -1)
        encoder_feature = encoder_outputs.view(-1, self.hidden_dim * 2)
        
        # ⭐ Create padding mask cho internal memory (all positions valid)
        enc_padding_mask = torch.ones(batch_size, self.internal_memory.size(0), device=y_t_1.device)
        
        # ⭐ Inference warm-up cho closed-book
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim),
                                 c_decoder.view(-1, self.hidden_dim)), 1)
            c_t, _, coverage_next = self.memory_attention(s_t_hat, encoder_outputs, encoder_feature,
                                                         enc_padding_mask, coverage)
            coverage = coverage_next

        # ⭐ Standard decoder forward
        y_t_1_embd = self.dropout(self.embedding(y_t_1))
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.reshape(-1, self.hidden_dim),
                             c_decoder.reshape(-1, self.hidden_dim)), 1)

        # ⭐ Memory attention - attend to internal memory
        c_t, attn_dist, coverage_next = self.memory_attention(s_t_hat, encoder_outputs, encoder_feature,
                                                             enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        # ⭐ Pointer-generator mechanism
        p_gen = None
        if self.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        # ⭐ Output generation
        output = torch.cat((lstm_out.reshape(-1, self.hidden_dim), c_t), 1)
        output = self.out1(output)
        output = F.relu(output)
        output = self.out2(output)
        vocab_dist = F.softmax(output, dim=1)

        # ⭐ Final distribution với pointer-generator
        if self.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

@META_ARCHITECTURE.register()
class Closedbook(nn.Module):
    def __init__(self, config, vocab: Vocab, model_file_path=None, is_eval=False):
        super(Closedbook, self).__init__()
        
        self.encoder = Encoder(config, vocab)
        self.decoder = Decoder(config, vocab)
        self.closedbook_decoder = ClosedbookDecoder(config, vocab)
        self.reduce_state = ReduceState(config, vocab)
        self.vocab = vocab
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu else 'cpu')
        self.MAX_LEN = vocab.max_sentence_length
        self.max_src_len = vocab.max_sentence_length

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.closedbook_decoder.load_state_dict(state.get('closedbook_decoder_state_dict', {}), strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])

    def forward(self, src, trg, gamma=0.5):
        """Hybrid training theo paper"""
        batch_size = src.size(0)
        src_len = src.size(1)
        trg_len = trg.size(1)
        
        # Encoder processing
        seq_lens = [(src[i] != self.vocab.pad_idx).sum().item() for i in range(batch_size)]
        seq_lens = sorted(seq_lens, reverse=True)
        
        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(src, seq_lens)
        decoder_hidden = self.reduce_state(encoder_hidden)
        enc_padding_mask = (src != self.vocab.pad_idx).float()
        
        # Initialize states
        s_t_attn = decoder_hidden
        s_t_cb = (torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device),
                  torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device))
        c_t_attn = torch.zeros(batch_size, self.config.hidden_dim * 2, device=self.device)
        c_t_cb = torch.zeros(batch_size, self.config.hidden_dim * 2, device=self.device)
        coverage_attn = torch.zeros(batch_size, src_len, device=self.device)
        coverage_cb = torch.zeros(batch_size, self.max_src_len, device=self.device)

        # Pointer-generator setup
        extra_zeros = None
        enc_batch_extend_vocab = None
        if self.config.pointer_gen:
            enc_batch_extend_vocab = src.clone()
            if src.size(1) < self.vocab.vocab_size:
                extra_zeros = torch.zeros(batch_size, self.vocab.vocab_size - src.size(1), device=self.device)

        # Hybrid decoding
        y_t_1 = torch.full((batch_size,), self.vocab.bos_idx, dtype=torch.long, device=self.device)
        step_losses = []
        attention_outputs = []
        
        for t in range(trg_len):
            # Attention decoder step
            final_dist_attn, s_t_attn, c_t_attn, attn_dist, p_gen, coverage_attn = self.decoder(
                y_t_1, s_t_attn, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_attn, extra_zeros, enc_batch_extend_vocab, coverage_attn, t
            )
            
            # Closed-book decoder step
            final_dist_cb, s_t_cb, c_t_cb, attn_dist_cb, p_gen_cb, coverage_cb = self.closedbook_decoder(
                y_t_1, s_t_cb, c_t_cb, extra_zeros, enc_batch_extend_vocab, coverage_cb, t
            )
            
            # Target and losses
            target_t = trg[:, t]
            attn_loss_t = F.cross_entropy(final_dist_attn, target_t, ignore_index=self.vocab.pad_idx)
            cb_loss_t = F.cross_entropy(final_dist_cb, target_t, ignore_index=self.vocab.pad_idx)
            
            # ⭐ Knowledge distillation loss
            kl_loss_t = F.kl_div(
                F.log_softmax(final_dist_cb, dim=-1),
                F.softmax(final_dist_attn.detach(), dim=-1),  # Detach attention outputs
                reduction='batchmean'
            )
            
            # Hybrid loss
            hybrid_loss_t = (1 - gamma) * attn_loss_t + gamma * (cb_loss_t + 0.1 * kl_loss_t)
            step_losses.append(hybrid_loss_t)
            attention_outputs.append(final_dist_attn)
            
            # Teacher forcing
            y_t_1 = target_t

        hybrid_loss = torch.stack(step_losses).mean()
        outputs = torch.stack(attention_outputs, dim=1)
        
        return outputs, hybrid_loss

    def forward_closed_book(self, trg):
        """Pure closed-book training"""
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        
        # Initialize for closed-book only
        s_t = (torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device),
               torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device))
        c_t = torch.zeros(batch_size, self.config.hidden_dim * 2, device=self.device)
        coverage = torch.zeros(batch_size, self.max_src_len, device=self.device)
        
        extra_zeros = None
        enc_batch_extend_vocab = None
        if self.config.pointer_gen:
            extra_zeros = torch.zeros(batch_size, self.vocab.vocab_size, device=self.device)
            enc_batch_extend_vocab = torch.zeros(batch_size, self.max_src_len, dtype=torch.long, device=self.device)

        decoder_outputs = []
        y_t_1 = torch.full((batch_size,), self.vocab.bos_idx, dtype=torch.long, device=self.device)
        
        for t in range(trg_len):
            final_dist, s_t, c_t, attn_dist, p_gen, coverage = self.closedbook_decoder(
                y_t_1, s_t, c_t, extra_zeros, enc_batch_extend_vocab, coverage, t
            )
            decoder_outputs.append(final_dist)
            y_t_1 = trg[:, t]
        
        outputs = torch.stack(decoder_outputs, dim=1)
        outputs_reshaped = outputs.view(-1, outputs.size(-1))
        trg_reshaped = trg.view(-1)
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx)
        loss = loss_fn(outputs_reshaped, trg_reshaped)
        
        return outputs, loss

    def predict(self, src: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
        """Hybrid prediction"""
        max_len = self.MAX_LEN

        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            src_len = src.size(1)
            
            # Encoder setup
            seq_lens = [(src[i] != self.vocab.pad_idx).sum().item() for i in range(batch_size)]
            seq_lens = sorted(seq_lens, reverse=True)
            
            encoder_outputs, encoder_feature, encoder_hidden = self.encoder(src, seq_lens)
            decoder_hidden = self.reduce_state(encoder_hidden)
            enc_padding_mask = (src != self.vocab.pad_idx).float()
            
            # Initialize states
            s_t_attn = decoder_hidden
            s_t_cb = (torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device),
                      torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device))
            c_t_attn = torch.zeros(batch_size, self.config.hidden_dim * 2, device=self.device)
            c_t_cb = torch.zeros(batch_size, self.hidden_size * 2, device=self.device)
            coverage_attn = torch.zeros(batch_size, src_len, device=self.device)
            coverage_cb = torch.zeros(batch_size, self.max_src_len, device=self.device)
            
            # Pointer-generator setup
            extra_zeros = None
            enc_batch_extend_vocab = None
            if self.pointer_gen:
                enc_batch_extend_vocab = src.clone()
                extra_zeros = torch.zeros(batch_size, self.vocab.vocab_size - src.size(1), device=self.device)

            # Generation
            y_t_1 = torch.full((batch_size,), self.vocab.bos_idx, dtype=torch.long, device=self.device)
            generated_tokens = []
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            
            for t in range(max_len):
                final_dist_attn, s_t_attn, c_t_attn, _, _, coverage_attn = self.decoder(
                    y_t_1, s_t_attn, encoder_outputs, encoder_feature, enc_padding_mask,
                    c_t_attn, extra_zeros, enc_batch_extend_vocab, coverage_attn, t
                )
                
                final_dist_cb, s_t_cb, c_t_cb, _, _, coverage_cb = self.closedbook_decoder(
                    y_t_1, s_t_cb, c_t_cb, extra_zeros, enc_batch_extend_vocab, coverage_cb, t
                )
                
                # Combine predictions
                combined_dist = (1 - gamma) * final_dist_attn + gamma * final_dist_cb
                next_token = combined_dist.argmax(dim=-1)
                generated_tokens.append(next_token)
                
                # Check EOS
                current_eos_generated = (next_token == self.vocab.eos_idx)
                finished_sequences = finished_sequences | current_eos_generated
                
                if finished_sequences.all():
                    break
                
                y_t_1 = next_token
            
            outputs = torch.stack(generated_tokens, dim=1)
            return outputs

    def predict_closed_book(self, max_len: int = None) -> torch.Tensor:
        """Pure closed-book prediction"""
        if max_len is None:
            max_len = 50
        
        self.eval()
        with torch.no_grad():
            batch_size = 1
            
            s_t = (torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device),
                   torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device))
            c_t = torch.zeros(batch_size, self.config.hidden_dim * 2, device=self.device)
            coverage = torch.zeros(batch_size, self.max_src_len, device=self.device)
            
            extra_zeros = None
            enc_batch_extend_vocab = None
            if self.config.pointer_gen:
                extra_zeros = torch.zeros(batch_size, self.vocab.vocab_size, device=self.device)
                enc_batch_extend_vocab = torch.zeros(batch_size, self.max_src_len, dtype=torch.long, device=self.device)

            y_t_1 = torch.full((batch_size,), self.vocab.bos_idx, dtype=torch.long, device=self.device)
            generated_tokens = []
            
            for t in range(max_len):
                final_dist, s_t, c_t, _, _, coverage = self.closedbook_decoder(
                    y_t_1, s_t, c_t, extra_zeros, enc_batch_extend_vocab, coverage, t
                )
                
                next_token = final_dist.argmax(dim=-1)
                generated_tokens.append(next_token)
                
                if next_token.item() == self.vocab.eos_idx:
                    break
                
                y_t_1 = next_token
            
            outputs = torch.stack(generated_tokens, dim=1)
            return outputs


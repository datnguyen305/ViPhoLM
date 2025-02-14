# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from builders.model_builder import META_ARCHITECTURE

from numpy import random
from models.pointer_generator.layers import Encoder
from models.pointer_generator.layers import Decoder
from models.pointer_generator.layers import ReduceState

from vocabs.vocab import Vocab


random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

@META_ARCHITECTURE.register()
class PointerGeneratorModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(PointerGeneratorModel, self).__init__()
        self.vocab = vocab
        self.config = config
        self.d_model = config.d_model

        encoder = Encoder(config.encoder, vocab)
        decoder = Decoder(config.decoder, vocab)
        reduce_state = ReduceState(config.reduce_state)

        self.MAX_LENGTH = vocab.max_sentence_length + 2

        # shared the embedding between encoder and decoder
        decoder.tgt_word_emb.weight = encoder.src_word_emb.weight

        if config.is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if config.use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        config = self.config
        vocab = self.vocab

        x_lens = (x != vocab.pad_idx).sum(dim=1)

        # Sắp xếp x theo độ dài giảm dần
        x_lens, sort_indices = torch.sort(x_lens, descending=True)
        x = x[sort_indices]
        labels = labels[sort_indices] 

        # Tạo enc_padding_mask (đảm bảo không phải None)
        enc_padding_mask = (x != vocab.pad_idx).float()

        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(x, x_lens)
        decoder_hidden = self.reduce_state(encoder_hidden)
        
        batch_size = x.size(0)
        decoder_input = torch.full((batch_size, 1), self.vocab.bos_idx, dtype=torch.long, device=x.device)
    
        c_t = torch.zeros(batch_size, config.hidden_dim * 2, device=x.device)
        coverage = None if not config.is_coverage else torch.zeros_like(x, dtype=torch.float)
        
        loss = 0
        for t in range(labels.size(1)):
            final_dist, decoder_hidden, c_t, _, _, coverage = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_feature,
                enc_padding_mask, c_t, None, x, coverage, t
            )
            
            target = labels[:, t]
            log_probs = torch.log(final_dist + 1e-12)
            loss += F.nll_loss(log_probs, target, ignore_index=self.vocab.pad_idx)
            
            decoder_input = target.unsqueeze(1)
        
        return loss / batch_size

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        config = self.config
        vocab = self.vocab

        x_lens = (x != self.vocab.pad_idx).sum(dim=1)  # Chiều dài chuỗi

        # Tạo encoder inputs
        batch_size = x.size(0)
        max_enc_len = x_lens.max().item()

        # Tạo encoder position (enc_pos)
        enc_pos = torch.zeros((batch_size, max_enc_len), dtype=torch.long, device=x.device)
        for i in range(batch_size):
            enc_pos[i, :x_lens[i]] = torch.arange(1, x_lens[i] + 1, dtype=torch.long, device=x.device)

        enc_padding_mask = (x != vocab.pad_idx).float()  # Mặt nạ padding cho encoder

        # Tạo các tensor cần thiết cho pointer-generator (nếu có)
        enc_batch_extend_vocab = None
        extra_zeros = None
        if config.pointer_gen:
            enc_batch_extend_vocab = torch.zeros((batch_size, max_enc_len), dtype=torch.long, device=x.device)
            if hasattr(x, 'max_art_oovs') and x.max_art_oovs > 0:
                extra_zeros = torch.zeros((batch_size, x.max_art_oovs), dtype=torch.float, device=x.device)

        # Khởi tạo c_t (context vector)
        c_t = torch.zeros((batch_size, 2 * config.hidden_dim), dtype=torch.float, device=x.device)

        # Tạo coverage nếu cần thiết (nếu sử dụng coverage mechanism)
        coverage = None
        if config.is_coverage:
            coverage = torch.zeros((batch_size, max_enc_len), dtype=torch.float, device=x.device)

        # Encoder output
        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(x, x_lens)

        # Reduce state từ encoder hidden state
        decoder_hidden = self.reduce_state(encoder_hidden)

        # Khởi tạo decoder input (sử dụng BOS token)
        decoder_input = torch.full((batch_size, 1), self.vocab.bos_idx, dtype=torch.long, device=x.device)

        # Lặp qua từng bước của decoder để sinh ra output
        outputs = []
        for t in range(self.MAX_LENGTH):
            final_dist, decoder_hidden, c_t, attn_dist, p_gen, next_coverage = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_feature,
                enc_padding_mask, c_t, extra_zeros, enc_batch_extend_vocab, coverage, t
            )

            top_idx = final_dist.argmax(dim=-1)  # Lấy chỉ số từ output distribution
            outputs.append(top_idx)

            decoder_input = top_idx.unsqueeze(1)  # Cập nhật decoder input cho bước tiếp theo

            if (top_idx == self.vocab.eos_idx).all():  # Dừng nếu gặp EOS token
                break

        return torch.cat(outputs, dim=1)  # Trả về kết quả dự đoán

import os
import torch
import torch.nn as nn
from vocabs.vocab import Vocab
from .module import Encoder, Decoder
from builders.model_builder import META_ARCHITECTURE


class ClosedBook(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(ClosedBook, self).__init__()
        self.config = config
        self.vocab = vocab
        self.device = config['device']
        self._init_vocab_dict()

    def _init_vocab_dict(self):
        self.idx2token = self.vocab.stoi
        self.max_vocab_size = self.vocab.vocab_size
        self.padding_token_idx = self.vocab.pad_idx
        self.unknown_token_idx = self.vocab.unk_idx
        self.sos_token_idx = self.vocab.bos_idx
        self.eos_token_idx = self.vocab.eos_idx

@META_ARCHITECTURE.register
class ClosedBookSummarization(ClosedBook):
    def __init__(self, config, vocab: Vocab):
        super().__init__(config, vocab)

        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_enc_layers = config.num_enc_layers
        self.num_dec_layers = config.num_dec_layers
        self.bidirectional = config.bidirectional
        self.dropout_ratio = config.dropout_ratio
        self.target_max_length = self.vocab.max_sentence_length + 2  # +2 for <bos> and <eos>
        self.d_model = config.d_model
        self.is_attention = config.is_attention
        self.is_pgen = config.is_pgen and self.is_attention
        self.is_coverage = config.is_coverage and self.is_attention
        if self.is_coverage:
            self.cov_loss_lambda = config.cov_loss_lambda

        self.context_size = self.hidden_size

        self.source_token_embedder = nn.Embedding(self.max_vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        self.target_token_embedder = nn.Embedding(self.max_vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config, self.vocab)

    def encode(self, source_idx, source_length):
        source_idx_embed = source_idx.clone()
        source_idx_embed[source_idx_embed >= self.max_vocab_size] = self.unknown_token_idx
        source_embeddings = self.source_token_embedder(source_idx_embed)
        encoder_outputs, encoder_hidden_states = self.encoder(source_embeddings, source_length)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]

        encoder_hidden_states = (encoder_hidden_states[0][::2], encoder_hidden_states[1][::2])

        return encoder_outputs, encoder_hidden_states

    def predict(self, input_ids, extended_source_idx, extra_zeros):
        """
        Dự đoán theo batch (B=1), sử dụng greedy search.
        Trả về một tensor các indices (dãy số).
        """
        
        # 1. Encode (Batched)
        source_idx = input_ids
        # Tính độ dài từ padding_token_idx
        source_length = (torch.ne(source_idx, self.padding_token_idx).sum(dim=1)).long()
        encoder_outputs, decoder_hidden_states = self.encode(source_idx, source_length)

        batch_size = source_idx.size(0) # Sẽ là 1
        src_len = source_idx.size(1)

        # 2. Khởi tạo kwargs (Batched) - Giống như trong 'forward'
        kwargs = {}
        if self.is_attention:
            kwargs['encoder_outputs'] = encoder_outputs
            kwargs['encoder_masks'] = torch.ne(source_idx, self.padding_token_idx).to(self.device)
            kwargs['context'] = torch.zeros((batch_size, 1, self.context_size)).to(self.device)
        if self.is_pgen:
            kwargs['extra_zeros'] = extra_zeros
            kwargs['extended_source_idx'] = extended_source_idx
        if self.is_coverage:
            kwargs['coverages'] = torch.zeros((batch_size, 1, src_len)).to(self.device)

        # 3. Khởi tạo vòng lặp Greedy (Batched)
        all_generated_indices = []
        # Tensor theo dõi các câu đã kết thúc (chỉ có 1 câu)
        is_finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        
        # Input đầu tiên: [SOS]
        # (Vì batch_size=1, chúng ta dùng [[self.sos_token_idx]])
        input_target_idx = torch.LongTensor([[self.sos_token_idx]] * batch_size).to(self.device) # Shape: (1, 1)

        for _ in range(self.target_max_length):
            # Map OOV indices (>= vocab_size) về UNK (unknown_token_idx)
            # để có thể tra cứu embedding
            input_target_idx_embed = input_target_idx.clone()
            input_target_idx_embed[input_target_idx_embed >= self.max_vocab_size] = self.unknown_token_idx
            
            input_embeddings = self.target_token_embedder(input_target_idx_embed) # Shape: (1, 1, E)

            # Chạy 1 bước Decoder
            vocab_dists, decoder_hidden_states, kwargs = self.decoder(
                input_embeddings, decoder_hidden_states, kwargs=kwargs
            ) # vocab_dists shape: (1, 1, V_extended)

            # Greedy search: Lấy token có xác suất cao nhất
            word_id = torch.argmax(vocab_dists, dim=2) # Shape: (1, 1)
            word_id_squeezed = word_id.squeeze(1) # Shape: (1)

            # --- Quản lý việc dừng (EOS) ---
            # 1. Lấy token thực sự (PAD nếu đã xong, nếu không thì lấy word_id)
            current_output_tokens = word_id_squeezed.masked_fill(is_finished, self.padding_token_idx)
            
            # 2. Kiểm tra xem câu nào *mới* kết thúc ở bước này
            newly_finished = (~is_finished) & (current_output_tokens == self.eos_token_idx)
            
            # 3. Cập nhật is_finished cho các bước tiếp theo
            is_finished = is_finished | newly_finished
            
            # 4. Lưu trữ tensor index (Bao gồm cả EOS/PAD)
            all_generated_indices.append(current_output_tokens.unsqueeze(1))

            # 5. Dừng nếu tất cả các câu trong batch đều đã xong
            if is_finished.all(): # Sẽ dừng ngay khi câu duy nhất gặp EOS
                break

            # 6. Chuẩn bị input cho bước tiếp theo
            input_target_idx = word_id

        # 4. Gộp kết quả
        if all_generated_indices:
            generated_indices_tensor = torch.cat(all_generated_indices, dim=1) # Shape: (1, T)
        else:
            # Trường hợp đặc biệt (ví dụ: max_length=0)
            generated_indices_tensor = torch.empty(batch_size, 0, dtype=torch.long).to(self.device)

        # Trả về tensor index
        return generated_indices_tensor

    def forward(self, input_ids, labels, extended_source_idx, extra_zeros):
        # Encoder
        source_idx = input_ids
        source_length = (torch.ne(source_idx, self.padding_token_idx).sum(dim=1)).long()
        encoder_outputs, encoder_hidden_states = self.encode(source_idx, source_length)

        batch_size = source_idx.size(0)
        src_len = source_idx.size(1)

        # Decoder
        input_target_idx = labels[:, :-1] 
        output_target_idx = labels[:, 1:] 
        input_target_idx_embed = input_target_idx.clone()
        input_target_idx_embed[input_target_idx_embed >= self.max_vocab_size] = self.unknown_token_idx
        input_embeddings = self.target_token_embedder(input_target_idx_embed)  # B x dec_len x 128
        kwargs = {}
        if self.is_attention:
            kwargs['encoder_outputs'] = encoder_outputs  # B x src_len x 256
            kwargs['encoder_masks'] = torch.ne(source_idx, self.padding_token_idx).to(self.device)  # B x src_len
            kwargs['context'] = torch.zeros((batch_size, 1, self.context_size)).to(self.device)  # B x 1 x 256

        if self.is_pgen:
            kwargs['extra_zeros'] = extra_zeros  # B x max_oovs_num
            kwargs['extended_source_idx'] = extended_source_idx  # B x src_len

        if self.is_coverage:
            kwargs['coverages'] = torch.zeros((batch_size, 1, src_len)).to(self.device)  # B x 1 x src_len

        vocab_dists, _, kwargs = self.decoder(
            input_embeddings, encoder_hidden_states, kwargs=kwargs
        )
        # Loss
        probs_masks = torch.ne(output_target_idx, self.padding_token_idx)

        gold_probs = torch.gather(vocab_dists, 2, output_target_idx.unsqueeze(2)).squeeze(2)  # B x dec_len
        nll_loss = -torch.log(gold_probs + 1e-12)
        if self.is_coverage:
            coverage_loss = torch.sum(torch.min(kwargs['attn_dists'], kwargs['coverages']), dim=2)  # B x dec_len
            nll_loss = nll_loss + self.cov_loss_lambda * coverage_loss

        loss = nll_loss * probs_masks

        # Chia cho số lượng token thực tế (không phải padding)
        non_pad_tokens = probs_masks.sum(dim=1).float()
        # Tránh chia cho 0 nếu có câu rỗng
        non_pad_tokens = torch.where(non_pad_tokens == 0, 1.0, non_pad_tokens) 
        
        loss = loss.sum(dim=1) / non_pad_tokens
        loss = loss.mean()
        # --- KẾT THÚC SỬA LỖI ---
        
        # Trả về (None, loss) để khớp với TextSumTaskOOV
        return None, loss
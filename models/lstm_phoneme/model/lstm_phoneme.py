


    


@META_ARCHITECTURE.register()
class LSTM_Model_Phoneme(nn.Module):
    def __init__(self, config, vocab: ViWordVocab):
        super().__init__()

        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2 # + 2 for bos and eos tokens
        self.d_model = config.d_model
        
        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)
        
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        encoder_outs, hidden_states = self.encoder(x)
    
        outs, _ = self.decoder(encoder_outs, hidden_states, labels)
        # outs: (B, S, 4, vocab_size)
        loss_onset = self.loss(outs[:, :, 0, :].reshape(-1, self.vocab.vocab_size), labels[:, :, 0].reshape(-1))
        loss_medial = self.loss(outs[:, :, 1, :].reshape(-1, self.vocab.vocab_size), labels[:, :, 1].reshape(-1))
        loss_nucleus = self.loss(outs[:, :, 2, :].reshape(-1, self.vocab.vocab_size), labels[:, :, 2].reshape(-1))
        loss_coda = self.loss(outs[:, :, 3, :].reshape(-1, self.vocab.vocab_size), labels[:, :, 3].reshape(-1))

        loss = loss_onset + loss_medial + loss_nucleus + loss_coda
    
        return outs, loss
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs, encoder_states = self.encoder(x)
        batch_size = encoder_outputs.size(0)
        
        # --- 1. KHỞI TẠO INPUT ĐÚNG ---
        # Input phải có shape (Batch, 1, 4).
        # Mặc định điền PAD, riêng vị trí 0 (âm đầu) điền BOS.
        decoder_input = torch.full((batch_size, 1, 4), self.vocab.pad_idx, dtype=torch.long, device=x.device)
        decoder_input[:, :, 0] = self.vocab.bos_idx 
        
        decoder_hidden, decoder_memory = encoder_states
        outputs = []
        
        # --- 2. VÒNG LẶP SINH TỪ ---
        for i in range(self.MAX_LENGTH):
            # Chạy 1 bước decoder
            # decoder_output shape: (B, 1, 4, vocab_size)
            decoder_output, (decoder_hidden, decoder_memory) = self.decoder.forward_step(decoder_input, (decoder_hidden, decoder_memory))
            
            # Chọn từ có xác suất cao nhất (Greedy)
            # decoder_input mới sẽ có shape: (B, 1, 4) -> Làm input cho vòng lặp sau
            decoder_input = decoder_output.argmax(dim=-1)
            outputs.append(decoder_input)
            
            # --- 3. ĐIỀU KIỆN DỪNG ---
            # Chỉ áp dụng break sớm nếu chạy 1 câu (Batch size = 1)
            # Nếu Batch > 1, ta cứ chạy hết MAX_LENGTH để đảm bảo tensor đồng nhất (hoặc phải dùng padding mask phức tạp hơn)
            if batch_size == 1:
                # Kiểm tra xem âm đầu (index 0) có phải là EOS không
                token_onset = decoder_input[0, 0, 0].item()
                if token_onset == self.vocab.eos_idx and i >= 1:
                    break

            # Nối các bước lại: (B, Sequence_Length, 4)
            final_output = torch.cat(outputs, dim=1)
            
        return final_output
    
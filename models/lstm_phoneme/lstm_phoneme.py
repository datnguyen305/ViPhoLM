import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.viword_vocab import ViWordVocab
from builders.model_builder import META_ARCHITECTURE

class Encoder(nn.Module):
    def __init__(self, config, vocab: ViWordVocab):
        super().__init__()
        self.num_features = 4
        self.embedding = nn.ModuleList([
            nn.Embedding(vocab.vocab_size, config.hidden_size) 
            for _ in range(self.num_features)
        ])
        self.lstm = nn.LSTM(
            config.hidden_size * self.num_features,
            config.hidden_size,
            bidirectional=False,
            num_layers=config.layer_dim, 
            batch_first=True, 
            dropout=config.dropout
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        B, S, _  = input.size()
        embeds = []
        for i in range(self.num_features):
            embeds.append(self.dropout(self.embedding[i](input[:, :, i])))
            # embeds: (batch_size, seq_len, hidden_size) * 4
        embedded = torch.cat(embeds, dim=-1)
        # embedded: (batch_size, seq_len, hidden_size * 4)
        output, states = self.lstm(embedded)
        # output : (batch_size, seq_len, hidden_size)
        # states: (h_n, c_n) each: (num_layers, batch_size, hidden_size)

        return output, states
    
class Decoder(nn.Module):
    def __init__(self, config, vocab: ViWordVocab):
        super().__init__()

        self.vocab = vocab
        self.num_features = 4
        self.embedding = nn.ModuleList([
            nn.Embedding(vocab.vocab_size, config.hidden_size) 
            for _ in range(self.num_features)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            config.hidden_size * self.num_features,
            config.hidden_size, # Chưa thử để hidden_size * 4 
            bidirectional=False,
            num_layers=config.layer_dim, # 3 lstm 
            batch_first=True, 
            dropout=config.dropout
        )
        self.fc_onset = nn.Linear(config.hidden_size, vocab.vocab_size)
        self.fc_medial = nn.Linear(config.hidden_size, vocab.vocab_size)
        self.fc_nucleus = nn.Linear(config.hidden_size, vocab.vocab_size)
        self.fc_coda = nn.Linear(config.hidden_size, vocab.vocab_size)

    def forward(self, encoder_outputs: torch.Tensor, encoder_states: torch.Tensor, target_tensor: torch.Tensor):
        batch_size = encoder_outputs.size(0)
        # Initiate decoder's input [<BOS>, <PAD>, <PAD>, <PAD>]
        decoder_input = torch.empty(batch_size, 1, self.num_features, dtype=torch.long, device=encoder_outputs.device)
        for i in range(self.num_features):
            if i == 0: 
                decoder_input[:, :, i].fill_(self.vocab.bos_idx)
            else: 
                decoder_input[:, :, i].fill_(self.vocab.pad_idx)
        # decoder_input: (batch_size, 1, 4)

        decoder_hidden, decoder_memory = encoder_states
        decoder_outputs = []
        target_len = target_tensor.shape[1]
        # target_tensor: (batch_size, seq_len, 4)
        # target_tensor [: , i] (batch_size, 4)
        for i in range(target_len):
            decoder_output, (decoder_hidden, decoder_memory) = self.forward_step(decoder_input, (decoder_hidden, decoder_memory))
            # decoder_output: (B, 1, 4, vocab_size)
            decoder_outputs.append(decoder_output)

            # Teacher forcing: Feed the target as the next input
            decoder_input = target_tensor[:, i, :].unsqueeze(1) # Teacher forcing

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs: (B, S, 4, vocab_size)
        
        return decoder_outputs, (decoder_hidden, decoder_memory)

    def forward_step(self, input, states):
        embeds = []
        for i in range(self.num_features):
            embeds.append(self.dropout(self.embedding[i](input[:, :, i])))
        embedded = torch.cat(embeds, dim=-1)
        # embedded: (batch_size, 1, hidden_size * 4)
        output, (hidden, memory) = self.lstm(embedded, states)
        # output: (batch_size, 1, hidden_size)
        onset_out = self.fc_onset(output)
        medial_out = self.fc_medial(output)
        nucleus_out = self.fc_nucleus(output)
        coda_out = self.fc_coda(output)

        # *_out :(batch_size, 1, vocab_size)
        output = torch.stack([onset_out, medial_out, nucleus_out, coda_out], dim=2)
        # output: (B, 1, 4, vocab_size)

        return output, (hidden, memory)

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
        for _ in range(self.MAX_LENGTH):
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
                if token_onset == self.vocab.eos_idx:
                    break

            # Nối các bước lại: (B, Sequence_Length, 4)
            final_output = torch.cat(outputs, dim=1)
            
            return final_output
    
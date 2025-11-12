import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class EncoderHierarchical(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.lstm_word = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            bidirectional=True,
            num_layers=config.num_layers,
            batch_first=True
        )
        self.lstm_sent = nn.LSTM(
            config.hidden_size*2,
            config.hidden_size,
            bidirectional=True,
            num_layers=config.num_layers,          
            batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        """
        input_ids: (B, S, W) - batch_size, num_sentences, num_words
        returns:
            word_outputs: (B, S, W, H*2) - hidden states of word-level LSTM
            sent_embeds: (B, S, H*2) - sentence embeddings (from last word of each sentence)
            document_embedding: (1, B, H*2) - final document embedding
        """
        B, S, W = input_ids.size()
        
        # Flatten sentences for word-level LSTM
        input_flat = input_ids.view(B*S, W)                  # (B*S, W)
        embedded = self.embedding(input_flat)               # (B*S, W, H)
        
        word_outputs, (hidden_word, _) = self.lstm_word(embedded)  
        # word_outputs: (B*S, W, 2*H)
        # hidden_word : (2, B*S, H)
        
        # Sentence embedding: take last hidden state of word-level LSTM
        sent_embeds = torch.cat((hidden_word[0], hidden_word[1]), dim=-1)  # (B*S, 2*H)
        sent_embeds = sent_embeds.view(B, S, -1)           # (B, S, 2*H)
        
        # Sentence-level LSTM
        sent_outputs, (hidden_doc, _) = self.lstm_sent(sent_embeds) 
        # sent_outputs: (B, S, 2*H)
        # hidden_doc : (2, B, H)
        
        # Document embedding: concatenate last forward + backward hidden
        hidden_doc = torch.cat([hidden_doc[0], hidden_doc[1]], dim=-1)  # (B, 2*H)
        document_embedding = hidden_doc
        
        # Word-level outputs reshaped for attention if needed
        word_outputs = word_outputs.view(B, S, W, -1)                    # (B, S, W, 2*H)
        
        # sent_outputs: (B, S, 2*H), document_embedding: (B, 2*H)
        return sent_outputs, document_embedding

class Attention(nn.Module):
    """
    Tính Attention (Bahdanau-style)
    score(s_t, h_i) = v^T * tanh(W_s * s_t + W_h * h_i)
    """
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        # encoder_hidden_dim = H*2 (từ Bi-LSTM)
        # decoder_hidden_dim = H (từ Uni-LSTM)
        
        self.W_h = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False) # Chiếu h_i
        self.W_s = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False) # Chiếu s_t
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden (s_t): (B, dec_H)
        # encoder_outputs (h_i): (B, S, enc_H)
        
        B, S, _ = encoder_outputs.size()
        
        # (B, 1, dec_H)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)
        
        # Tính energy
        # W_h * h_i -> (B, S, dec_H)
        # W_s * s_t -> (B, 1, dec_H)
        # Tanh(...) -> (B, S, dec_H)
        energy = torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_hidden_expanded))
        
        # v * Tanh(...) -> (B, S, 1)
        attn_scores = self.v(energy).squeeze(2) # (B, S)
        
        # Trọng số
        attn_weights = F.softmax(attn_scores, dim=1) # (B, S)
        
        # Context vector c_t
        # (B, 1, S) bmm (B, S, enc_H) -> (B, 1, enc_H)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        return context_vector.squeeze(1), attn_weights # (B, enc_H), (B, S)

class DecoderWordLevelWithAttention(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx=vocab.pad_idx)
        
        encoder_hidden_dim = config.hidden_size * 2
        decoder_hidden_dim = config.hidden_size 
        
        self.attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
        self.lstm = nn.LSTM(
            config.hidden_size + encoder_hidden_dim,
            decoder_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(decoder_hidden_dim + encoder_hidden_dim, vocab.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward_step(self, decoder_input, hidden, cell, encoder_outputs):
        """
        Chạy 1 bước decode (1 token)
        decoder_input: (B, 1)
        hidden, cell: (1, B, H)
        encoder_outputs: (B, S, 2H)
        """
        embedded = self.embedding(decoder_input)  # (B, 1, H)
        embedded = self.dropout(embedded)

        # Attention
        context_vector, attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)  # (B, 2H)

        # Chuẩn bị input cho LSTM
        lstm_input = torch.cat((embedded, context_vector.unsqueeze(1)), dim=2)  # (B, 1, H+2H)

        # Chạy 1 bước LSTM
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))  # (B, 1, H)

        # Dự đoán từ
        output_for_pred = torch.cat((lstm_output.squeeze(1), context_vector), dim=1)  # (B, H+2H)
        prediction = self.out(output_for_pred)  # (B, V)

        return prediction, hidden, cell, attn_weights

    def forward(self, target_flat, initial_hidden, initial_cell, encoder_outputs):
        """
        Huấn luyện với Teacher Forcing
        """
        B, T = target_flat.size()
        decoder_input = torch.full((B, 1), self.vocab.bos_idx,
                                   dtype=torch.long, device=target_flat.device)
        hidden, cell = initial_hidden, initial_cell
        outputs = []

        for t in range(T):
            prediction, hidden, cell, _ = self.forward_step(
                decoder_input, hidden, cell, encoder_outputs
            )
            outputs.append(prediction)
            decoder_input = target_flat[:, t].unsqueeze(1)  # teacher forcing

        outputs = torch.stack(outputs, dim=1)
        return outputs

@META_ARCHITECTURE.register()
class AbstractiveTXModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.config = config
        self.device = config.device
        
        self.encoder = EncoderHierarchical(config, vocab)
        self.decoder = DecoderWordLevelWithAttention(config, vocab)
        
        # Lớp Linear để biến đổi trạng thái cuối cùng của Encoder (2*H)
        # thành trạng thái ban đầu của Decoder (H)
        encoder_hidden_dim = config.hidden_size * 2
        decoder_hidden_dim = config.hidden_size
        self.fc_hidden = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        self.fc_cell = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        
        # Dùng CrossEntropyLoss chuẩn (Model 3 không dùng Pointer-Generator)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx)
        self.d_model = config.hidden_size
        self.to(self.device)
        self.max_length = vocab.max_sentence_length + 2
    def forward(self, src_hier, target_flat):
        """
        src_hier: (B, S, W) - Dữ liệu input phân cấp
        target_flat: (B, T) - Dữ liệu target (là input đã làm phẳng)
        """
        
        # 1. Encode
        # sent_outputs (h_i): (B, S, 2*H)
        # doc_embedding (e_D): (B, 2*H)
        sent_outputs, doc_embedding = self.encoder(src_hier)
        
        # 2. Khởi tạo Decoder
        # Biến đổi e_D (2*H) thành s_0 (H)
        initial_hidden = torch.tanh(self.fc_hidden(doc_embedding)).unsqueeze(0) # (1, B, H)
        initial_cell = torch.tanh(self.fc_cell(doc_embedding)).unsqueeze(0)   # (1, B, H)
        
        # 3. Decode
        # decoder_outputs: (B, T, V)
        decoder_outputs = self.decoder(
            target_flat=target_flat,
            initial_hidden=initial_hidden,
            initial_cell=initial_cell,
            encoder_outputs=sent_outputs
        )
        
        # 4. Tính Loss
        loss = self.loss_fn(
            decoder_outputs.view(-1, self.vocab.vocab_size), # (B*T, V)
            target_flat.view(-1)                             # (B*T)
        )
        
        return decoder_outputs, loss

    def predict(self, src_hier: torch.Tensor):
        """
        Tạo văn bản (Greedy decoding)
        """
        self.eval()
        B = src_hier.size(0)

        with torch.no_grad():
            sent_outputs, doc_embedding = self.encoder(src_hier)
            hidden = torch.tanh(self.fc_hidden(doc_embedding)).unsqueeze(0)
            cell = torch.tanh(self.fc_cell(doc_embedding)).unsqueeze(0)

            decoder_input = torch.full((B, 1), self.vocab.bos_idx,
                                       dtype=torch.long, device=self.device)
            outputs = []

            for _ in range(self.max_length):
                # 🔹 Gọi 1 bước forward_step
                prediction, hidden, cell, _ = self.decoder.forward_step(
                    decoder_input, hidden, cell, sent_outputs
                )

                # 🔹 Greedy chọn token
                decoder_input = prediction.argmax(dim=-1).unsqueeze(1)  # (B, 1)
                outputs.append(decoder_input)

                # 🔹 Dừng nếu batch size = 1 và gặp <EOS>
                if B == 1 and decoder_input.item() == self.vocab.eos_idx:
                    break

            outputs = torch.cat(outputs, dim=1)
        return outputs

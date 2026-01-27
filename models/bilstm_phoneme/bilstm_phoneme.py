import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.viword_vocab import ViWordVocab
from builders.model_builder import META_ARCHITECTURE

class Encoder(nn.Module):
    def __init__(self, config, vocab: ViWordVocab):
        super().__init__()
        self.num_features = 4
        self.num_layer = config.layer_dim
        self.embedding = nn.ModuleList([
            nn.Embedding(vocab.vocab_size, config.hidden_size) 
            for _ in range(self.num_features)
        ])

        self.lstm = nn.LSTM(
            config.hidden_size * self.num_features,
            config.hidden_size,
            bidirectional=True,
            num_layers=config.layer_dim, 
            batch_first=True, 
            dropout=config.dropout
        )
        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.li_hid = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.li_cn = nn.Linear(config.hidden_size * 2, config.hidden_size) 

    def reshape_encoder_states(self, hidden, cell):
        hidden = hidden.reshape(self.num_layer, 2, hidden.shape[1], hidden.shape[2])
        cell = cell.reshape(self.num_layer, 2, cell.shape[1], cell.shape[2])
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
        # hidden, cell: (num_layers, batch_size, hidden_size * 2)
        return hidden, cell

    def forward(self, input):
        B, S, _  = input.size()
        embeds = []
        for i in range(self.num_features):
            embeds.append(self.dropout(self.embedding[i](input[:, :, i])))
            # embeds: (batch_size, seq_len, hidden_size) * 4
        embedded = torch.cat(embeds, dim=-1)
        # embedded: (batch_size, seq_len, hidden_size * 4)
        output, states = self.lstm(embedded)
        # output : (batch_size, seq_len, hidden_size * 2)
        # states: (h_n, c_n) each: (num_layers, batch_size, hidden_size)

        output = self.out(output)
        # output: (batch_size, seq_len, hidden_size)

        h_n, c_n = states
        h_n, c_n = self.reshape_encoder_states(h_n, c_n)
        h_n = self.li_hid(h_n)
        c_n = self.li_cn(c_n)

        states = (h_n, c_n)

        """
            output shape: (batch_size, seq_len, hidden_size)
            states: (h_n, c_n) each: (num_layers, batch_size, hidden_size)
        """
        return output, states



class BahdanauAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.W2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.V = nn.Linear(config.hidden_size, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Inputs:
        encoder_outputs: (B, S, hidden_size)
        decoder_hidden: (num_layers, B, hidden_size)

        Returns:
        context: (B, 1, hidden_size)
        alphas: (B, S)
        """

        decoder_hidden = decoder_hidden[-1]
        # decoder_hidden: (B, hidden_size)

        # Additive attention
        scores = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden.unsqueeze(1)))).squeeze(-1)
        # (B, S, hidden_size) + (B, 1, hidden_size) -> (B, S, hidden_size) -> (B, S)
        # Attention weights
        alphas = F.softmax(scores, dim=-1)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas.unsqueeze(1), encoder_outputs)
        # (B, 1, S) bmm (B, S, hidden_size) -> (B, 1, hidden_size)

        # context shape: [B, 1, hidden_size], alphas shape: [B, S]
        return context, alphas

class Decoder(nn.Module):
    def __init__(self, config, vocab: ViWordVocab):
        super().__init__()

        self.vocab = vocab
        self.num_features = 4
        
        # 1. Lấy cờ bật/tắt attention (Mặc định là True nếu không khai báo)
        self.use_attention = getattr(config, 'use_attention', True) 

        self.embedding = nn.ModuleList([
            nn.Embedding(vocab.vocab_size, config.hidden_size) 
            for _ in range(self.num_features)
        ])
        self.dropout = nn.Dropout(config.dropout)

        # 2. Tính toán kích thước đầu vào cho LSTM
        # Mặc định: Embedding (features * hidden)
        lstm_input_size = config.hidden_size * self.num_features

        # Nếu dùng Attention: input = Embedding + Context Vector (hidden_size)
        if self.use_attention:
            self.attention = BahdanauAttention(config)
            lstm_input_size += config.hidden_size 
        else:
            self.attention = None
            # Nếu KHÔNG dùng attention, input chỉ là Embedding thôi

        self.lstm = nn.LSTM(
            lstm_input_size, 
            config.hidden_size, 
            bidirectional=False,
            num_layers=config.layer_dim, 
            batch_first=True, 
            dropout=config.dropout
        )
        
        self.fc_onset = nn.Linear(config.hidden_size, vocab.vocab_size)
        self.fc_medial = nn.Linear(config.hidden_size, vocab.vocab_size)
        self.fc_nucleus = nn.Linear(config.hidden_size, vocab.vocab_size)
        self.fc_coda = nn.Linear(config.hidden_size, vocab.vocab_size)

    def forward(self, encoder_outputs: torch.Tensor, encoder_states: torch.Tensor, target_tensor: torch.Tensor):
        # Hàm forward giữ nguyên, không cần sửa gì
        # Vì logic xử lý từng bước nằm trong forward_step
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, self.num_features, dtype=torch.long, device=encoder_outputs.device)
        for i in range(self.num_features):
            if i == 0: 
                decoder_input[:, :, i].fill_(self.vocab.bos_idx)
            else: 
                decoder_input[:, :, i].fill_(self.vocab.pad_idx)
        
        decoder_hidden, decoder_memory = encoder_states
        decoder_outputs = []
        target_len = target_tensor.shape[1]

        for i in range(target_len):
            decoder_output, (decoder_hidden, decoder_memory) = self.forward_step(
                decoder_input, (decoder_hidden, decoder_memory), encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            decoder_input = target_tensor[:, i, :].unsqueeze(1) 

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, (decoder_hidden, decoder_memory)

    def forward_step(self, input, states, encoder_outputs):
        embeds = []
        for i in range(self.num_features):
            embeds.append(self.dropout(self.embedding[i](input[:, :, i])))
        embedded = torch.cat(embeds, dim=-1)
        # embedded: (batch_size, 1, hidden_size * 4)

        # 3. Logic điều kiện Attention trong forward_step
        if self.use_attention:
            # Nếu bật: Tính context và nối vào input
            context, _ = self.attention(encoder_outputs, states[0])
            # context: (B, 1, hidden_size)
            embedded = torch.cat((embedded, context), dim=-1)
            # embedded: (B, 1, hidden_size * 5)
        
        # Nếu tắt: giữ nguyên embedded (B, 1, hidden_size * 4) và đưa thẳng vào LSTM

        output, (hidden, memory) = self.lstm(embedded, states)
        
        onset_out = self.fc_onset(output)
        medial_out = self.fc_medial(output)
        nucleus_out = self.fc_nucleus(output)
        coda_out = self.fc_coda(output)
        output = torch.stack([onset_out, medial_out, nucleus_out, coda_out], dim=2)

        return output, (hidden, memory)

@META_ARCHITECTURE.register()
class BiLSTM_Model_Phoneme(nn.Module):
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
        # targets: (B, S, 4)
        V = self.vocab.vocab_size
        loss = self.loss(outs.view(-1, V), labels.view(-1))
    
        return outs, loss
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs, encoder_states = self.encoder(x)
        batch_size = encoder_outputs.size(0)

        decoder_input = torch.full((batch_size, 1, 4), self.vocab.pad_idx, dtype=torch.long, device=x.device)
        decoder_input[:, :, 0] = self.vocab.bos_idx 
        
        decoder_hidden, decoder_memory = encoder_states
        outputs = []
        
        for _ in range(self.MAX_LENGTH):
            decoder_output, (decoder_hidden, decoder_memory) = self.decoder.forward_step(
                decoder_input, (decoder_hidden, decoder_memory), encoder_outputs
            )
            decoder_input = decoder_output.argmax(dim=-1)
            outputs.append(decoder_input)
            if batch_size == 1:
                token_onset = decoder_input[0, 0, 0].item()
                if token_onset == self.vocab.eos_idx:
                    break

        final_output = torch.cat(outputs, dim=1)
            
        return final_output
    
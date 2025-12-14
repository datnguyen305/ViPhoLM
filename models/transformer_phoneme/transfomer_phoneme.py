import torch
from torch import nn
from vocabs.viword_vocab import ViWordVocab
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class Transformer_Phoneme_Model(nn.Module):
    def __init__(self, config, vocab: ViWordVocab):
        super().__init__()

        self.src_pad_idx = vocab.pad_idx
        self.trg_pad_idx = vocab.pad_idx
        self.trg_bos_idx = vocab.bos_idx
        self.trg_eos_idx = vocab.eos_idx

        self.d_model = config.d_model
        self.device = config.device
        self.config = config
        self.vocab = vocab
        # 1. EMBEDDING: Input có 4 features (Onset, Medial, Nucleus, Coda)
        self.num_features = 4 
        
        # Tạo 4 lớp Embedding riêng cho ENCODER input
        self.src_embeddings = nn.ModuleList([
            nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=self.src_pad_idx)
            for _ in range(self.num_features)
        ])

        # Tạo 4 lớp Embedding riêng cho DECODER input (Target)
        self.trg_embeddings = nn.ModuleList([
            nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=self.trg_pad_idx)
            for _ in range(self.num_features)
        ])

        # Positional encoding
        self.pos_encoder = nn.Embedding(config.max_len, config.d_model)
        self.pos_decoder = nn.Embedding(config.max_len, config.d_model)

        # Encoder & Decoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_head,
            dim_feedforward=config.ffn_hidden, dropout=config.drop_prob, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model, nhead=config.n_head,
            dim_feedforward=config.ffn_hidden, dropout=config.drop_prob, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)

        # --- QUAN TRỌNG: OUTPUT LAYER ---
        # Thay vì 1 Linear layer, ta dùng 4 Linear layer để dự đoán 4 thành phần riêng biệt
        self.fc_onset = nn.Linear(config.d_model, vocab.vocab_size)
        self.fc_medial = nn.Linear(config.d_model, vocab.vocab_size)
        self.fc_nucleus = nn.Linear(config.d_model, vocab.vocab_size)
        self.fc_coda = nn.Linear(config.d_model, vocab.vocab_size)

        self.loss = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)

    # ------------------------- UTILS -------------------------
    def combine_embeddings(self, inputs, embedding_layers, pos_emb):
        x = 0
        for i in range(self.num_features):
            # Scale embedding by sqrt(d_model) is often used in Transformer paper 
            # to balance with positional encoding, but optional.
            # x += embedding_layers[i](inputs[:, :, i]) * math.sqrt(self.d_model)
            x += embedding_layers[i](inputs[:, :, i])
        
        seq_len = inputs.size(1)
        # Sửa: unsqueeze(0) để khớp batch dimension
        pos = pos_emb(torch.arange(seq_len, device=inputs.device)).unsqueeze(0)
        return x + pos

    def make_padding_mask(self, x):
        # x: [Batch, Len, 4]

        return (x == self.src_pad_idx).all(dim=-1)

    def make_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz, device=self.device) * float('-inf'), diagonal=1)

    # ------------------------- FORWARD -------------------------
    def forward(self, src, tgt):
        """
        src: [B, Src_Len, 4]
        tgt: [B, Tgt_Len, 4]
        """
        config = self.config
        
        # Trim length
        src = src[:, :config.max_len, :]
        tgt = tgt[:, :config.max_len, :]

        # Shift Target: Input [:-1], Label [1:]
        tgt_input = tgt[:, :-1, :] # [B, T-1, 4]
        tgt_output = tgt[:, 1:, :] # [B, T-1, 4] (Label thực tế)

        # Masks
        src_key_padding = self.make_padding_mask(src)
        tgt_key_padding = self.make_padding_mask(tgt_input)
        tgt_causal_mask = self.make_causal_mask(tgt_input.size(1))

        # Embedding & Forward
        enc_emb = self.combine_embeddings(src, self.src_embeddings, self.pos_encoder)
        dec_emb = self.combine_embeddings(tgt_input, self.trg_embeddings, self.pos_decoder)

        memory = self.encoder(enc_emb, src_key_padding_mask=src_key_padding)
        
        out = self.decoder(
            dec_emb, memory, 
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=src_key_padding
        ) # [B, T-1, D]

        # --- DỰ ĐOÁN 4 THÀNH PHẦN ---
        # Mỗi thành phần đi qua 1 linear layer riêng
        logit_onset = self.fc_onset(out)   # [B, T-1, V]
        logit_medial = self.fc_medial(out)
        logit_nucleus = self.fc_nucleus(out)
        logit_coda = self.fc_coda(out)

        # --- TÍNH LOSS TỔNG HỢP ---
        # Flatten để tính loss: [B*(T-1), V] vs [B*(T-1)]
        # Tách tgt_output thành 4 phần
        loss_onset = self.loss(logit_onset.reshape(-1, self.vocab.vocab_size), tgt_output[:, :, 0].reshape(-1))
        loss_medial = self.loss(logit_medial.reshape(-1, self.vocab.vocab_size), tgt_output[:, :, 1].reshape(-1))
        loss_nucleus = self.loss(logit_nucleus.reshape(-1, self.vocab.vocab_size), tgt_output[:, :, 2].reshape(-1))
        loss_coda = self.loss(logit_coda.reshape(-1, self.vocab.vocab_size), tgt_output[:, :, 3].reshape(-1))

        total_loss = loss_onset + loss_medial + loss_nucleus + loss_coda
        
        # Trả về logits (để debug nếu cần) và loss
        return (logit_onset, logit_medial, logit_nucleus, logit_coda), total_loss

    # ------------------------- PREDICT -------------------------
    def predict(self, src: torch.Tensor):
        self.eval()
        config = self.config
        
        # 1. Encode Source (Chỉ làm 1 lần)
        src = src[:, :config.max_len, :]
        src_key_padding = self.make_padding_mask(src)
        enc_emb = self.combine_embeddings(src, self.src_embeddings, self.pos_encoder)
        
        with torch.no_grad():
            memory = self.encoder(enc_emb, src_key_padding_mask=src_key_padding)
            
            # 2. Prepare Decoder Input
            bs = src.size(0)
            # Khởi tạo [BOS, PAD, PAD, PAD]
            tgt_seq = torch.full((bs, 1, 4), self.trg_pad_idx, device=self.device)
            tgt_seq[:, 0, 0] = self.trg_bos_idx 

            finished = torch.zeros(bs, dtype=torch.bool, device=self.device)

            for _ in range(config.max_decoding_len):
                # Tạo mask nhân quả (Causal Mask)
                tgt_causal_mask = self.make_causal_mask(tgt_seq.size(1))
                
                # Trong lúc predict, tgt_seq không có padding nên tgt_key_padding có thể bỏ qua hoặc để None
                # Tuy nhiên giữ lại cũng không sao nếu logic pad_idx chuẩn.
                
                dec_emb = self.combine_embeddings(tgt_seq, self.trg_embeddings, self.pos_decoder)
                
                out = self.decoder(
                    dec_emb, memory, 
                    tgt_mask=tgt_causal_mask,
                    # tgt_key_padding_mask=None, # Có thể để None để nhanh hơn
                    memory_key_padding_mask=src_key_padding
                )

                last_hidden = out[:, -1, :] # [B, D]

                # Greedy Search
                next_onset = self.fc_onset(last_hidden).argmax(dim=-1)
                next_medial = self.fc_medial(last_hidden).argmax(dim=-1)
                next_nucleus = self.fc_nucleus(last_hidden).argmax(dim=-1)
                next_coda = self.fc_coda(last_hidden).argmax(dim=-1)

                next_token = torch.stack([next_onset, next_medial, next_nucleus, next_coda], dim=1).unsqueeze(1)
                tgt_seq = torch.cat([tgt_seq, next_token], dim=1)

                # Check EOS tại vị trí Onset
                finished |= (next_onset == self.trg_eos_idx)
                if finished.all():
                    break
        
        return tgt_seq[:, 1:, :]
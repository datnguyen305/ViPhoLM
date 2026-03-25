import torch
from torch import nn
from vocabs.vocab_translation import MTVocab
from builders.model_builder import META_ARCHITECTURE
from models.transformer_phoneme_translation.utils.clone import clones
from models.transformer_phoneme_translation.utils.padding_mask import create_padding_mask, create_padding_mask_normal
from models.transformer_phoneme_translation.utils.causal_mask import create_causal_mask
from models.transformer_phoneme_translation.blocks.decoder_block import TransformerDecoderBlock
from models.transformer_phoneme_translation.blocks.encoder_block import TransformerEncoderBlock
from models.transformer_phoneme_translation.layers.phoneme_feed_forward import FeedForward
from models.transformer_phoneme_translation.embedding.positional_embedding import PositionalEncoding

@META_ARCHITECTURE.register()
class TransformerPhonemeTranslation(nn.Module):
    def __init__(self, config, vocab: MTVocab):
        super().__init__()
        self.vocab = vocab
        self.d_model = config.d_model
        self.MAX_LENGTH = vocab.vietnamese_max_sentence_length
        self.config = config

        # Positional Encoding
        self.PE = PositionalEncoding(self.d_model, max_len=self.config.max_len + 10)

        # Encoder English (1 feature)
        self.src_embedding = nn.Embedding(vocab.english_vocab_size, config.d_model)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoderBlock(config, self.vocab)

        # Decoder Vietnamese (3 features: Initial, Rhyme, Tone)
        self.num_features = 3 # Đặt chung là 3 cho phần Vietnamese
        self.linear = nn.Linear(config.d_model * self.num_features, config.d_model)
        self.decoder = TransformerDecoderBlock(config, self.vocab)

        
        self.tgt_embedding = clones(nn.Embedding(vocab.vietnamese_vocab_size, config.d_model), self.num_features)
        self.phoneme_ff = clones(FeedForward(config), self.num_features)
        self.outs = clones(nn.Linear(config.d_model, vocab.vietnamese_vocab_size), self.num_features)
        self.losses = clones(nn.CrossEntropyLoss(ignore_index=vocab.unk_idx), self.num_features)

    def forward(self, src, trg):
        # src: (B, S)
        # trg: (B, S, 3)
        src = src[:, :self.config.max_len] # Cắt nếu dài quá giới hạn
        trg = trg[:, :self.config.max_len]

        encoder_padding_mask = create_padding_mask_normal(src, 0)
        B, S = src.shape

        target = trg[:, 1:, :]
        # target: (B, S - 1, 3) [4, 5, 8, ... <eos>]
        
        decoder_input = trg[:, :-1, :]
        # decoder_input: (B, S - 1, 3) [<bos>, 3, 4, 5, 6, ...]
    
        # src: (B, S)
        
        # embeds = []
        # for i in range(self.num_features):
        #     embeds.append(self.dropout(self.src_embedding[i](src[:, :, i])))
        # # embeds: (B, S, d_model) * 3 
        # x = torch.cat(embeds, -1)
        
        x = self.dropout(self.src_embedding(src))
        # x: (B, S, d_model)
         
        # Positional Encoding
        x = self.PE(x)
        # x: (B, S, d_model)
        
        # Encoder
        memory = self.encoder(x, encoder_padding_mask) 
        # memory: (B, S, d_model)


        # Decoder
        B, S, W = decoder_input.shape
        decoder_padding_mask = create_padding_mask(decoder_input, 3)
        decoder_causal_mask = create_causal_mask(S, self.config.device)

        # decoder_input: (B, S, 3)
        embeds = []
        for i in range(self.num_features):
            embeds.append(self.dropout(self.tgt_embedding[i](decoder_input[:, :, i])))
        # embeds: (B, S, d_model) * 3 
        x = torch.cat(embeds, -1)
        # embeds: (B, S, d_model * 3)
        # x: (B, S, d_model * 3)
        x = self.linear(x)
        # x: (B, S, d_model)
         
        # Positional Encoding
        x = self.PE(x)
        # x: (B, S, d_model)

        x = self.decoder(x, memory, decoder_causal_mask, \
                         decoder_padding_mask, encoder_padding_mask)
        
        ff_out = []
        for i in range(self.num_features):
            ff_out.append(self.phoneme_ff[i](x))
        # ff_out: (B, S, d_model) * 3 

        ff_out = torch.stack(ff_out, -1)
        # ff_out: (B, S, d_model, 3)

        ff_prj = []
        for i in range(self.num_features):
            ff_prj.append(self.outs[i](ff_out[:, :, :, i]))
        # ff_out: (B, S, vocab_size) * 3 
        ff_prjout = torch.stack(ff_prj, -1)
        # ff_prjout: (B, S, vocab_size, 3)

        loss_result = []

        for i in range(self.num_features):
            loss_result.append(
                self.losses[i](
                    ff_prjout[:, :, :, i].reshape(-1, self.vocab.vietnamese_vocab_size), 
                    target[:, :, i].reshape(-1)
                )
            )
        # loss_result: List [loss_initial, loss_rhyme, loss_tone]
        total_loss = sum(loss_result)

        return 0, total_loss 
    
    def predict(self, src):
        self.eval()
        src = src[:, :self.config.max_len] # Cắt nếu dài quá giới hạn
        B = src.size(0)
        device = src.device
        
        encoder_padding_mask = create_padding_mask_normal(src, 0)
        x = self.dropout(self.src_embedding(src))
        x = self.PE(x)
        memory = self.encoder(x, encoder_padding_mask) 

        decoder_input = torch.full((B, 1, self.num_features), self.vocab.pad_idx, dtype=torch.long, device=device)
        decoder_input[:, 0, 0] = self.vocab.bos_idx 
        
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        outputs = []

        for _ in range(self.MAX_LENGTH): 
            embeds = []
            for i in range(self.num_features):
                embeds.append(self.dropout(self.tgt_embedding[i](decoder_input[:, :, i])))
            
            x = torch.cat(embeds, -1)
            x = self.linear(x)
            x = self.PE(x)

            trg_mask = create_padding_mask(decoder_input, 3)
            trg_causal_mask = create_causal_mask(decoder_input.size(1), device)

            x = self.decoder(x, memory, trg_causal_mask, trg_mask, encoder_padding_mask)
            x_last = x[:, -1:, :] # (B, 1, d_model)

            ff_outs = []
            for i in range(self.num_features):
                h = self.phoneme_ff[i](x_last)
                logit = self.outs[i](h)
                ff_outs.append(logit.argmax(dim=-1)) # (B, 1)
            
            # next_token: (B, 1, 3)
            next_token = torch.stack(ff_outs, dim=-1).squeeze(2)
            
            if finished.any():
                next_token[finished] = self.vocab.pad_idx
            
            outputs.append(next_token)
            
            finished |= (next_token[:, 0, 0] == self.vocab.eos_idx)

            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if finished.all():
                break
        
        # Kết quả: (B, Seq_Len, 3)
        return torch.cat(outputs, dim=1)
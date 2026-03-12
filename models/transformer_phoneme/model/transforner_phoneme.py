import torch
from torch import nn
from vocabs.viword_vocab import Vocab
from builders.model_builder import META_ARCHITECTURE
from models.transformer_phoneme.utils.clone import clones
from models.transformer_phoneme.utils.padding_mask import create_padding_mask
from models.transformer_phoneme.utils.causal_mask import create_causal_mask
from models.transformer_phoneme.blocks.decoder_block import TransformerDecoderBlock
from models.transformer_phoneme.blocks.encoder_block import TransformerEncoderBlock
from models.transformer_phoneme.layers.phoneme_feed_forward import FeedForward
from models.transformer_phoneme.embedding.positional_embedding import PositionalEncoding

@META_ARCHITECTURE.register()
class TransformerPhoneme(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.d_model = config.d_model
        self.MAX_LENGTH = vocab.max_sentence_length + 2
        self.config = config

        # Positional Encoding
        self.PE = PositionalEncoding(self.d_model, max_len=5000)

        # Encoder 
        self.num_features = 3 
        self.src_embedding = clones(nn.Embedding(vocab.vocab_size, config.d_model), self.num_features)
        self.linear = nn.Linear(config.d_model * 3, config.d_model)
        self.dropout = nn.Dropout(0.1)

        self.encoder = TransformerEncoderBlock(config, self.vocab)

        # Decoder  
        self.tgt_embedding = clones(nn.Embedding(vocab.vocab_size, config.d_model), self.num_features)
        self.decoder = TransformerDecoderBlock(config, self.vocab)
        self.phoneme_ff = clones(FeedForward(config), self.num_features)
        self.outs = clones(nn.Linear(config.d_model, vocab.vocab_size), self.num_features)
        self.losses = clones(nn.CrossEntropyLoss(ignore_index=vocab.unk_idx), self.num_features)

    def forward(self, src, trg):
        # src: (B, S, 3)
        # trg: (B, S, 3)

        encoder_padding_mask = create_padding_mask(src, 3)
        B, S, W = src.shape

        target = trg[:, 1:, :]
        # target: (B, S - 1, 3) [4, 5, 8, ... <eos>]
        
        decoder_input = trg[:, :-1, :]
        # decoder_input: (B, S - 1, 3) [<bos>, 3, 4, 5, 6, ...]
    
        # src: (B, S, 3)
        embeds = []
        for i in range(self.num_features):
            embeds.append(self.dropout(self.src_embedding[i](src[:, :, i])))
        # embeds: (B, S, d_model) * 3 
        x = torch.cat(embeds, -1)
        # embeds: (B, S, d_model * 3)
        # x: (B, S, d_model * 3)
        x = self.linear(x)
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
        decoder_causal_mask = create_causal_mask(S)

        # decoder_input: (B, S, 3)
        embeds = []
        for i in range(self.num_features):
            embeds.append(self.dropout(self.src_embedding[i](decoder_input[:, :, i])))
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
        ff_ff_prjout = torch.stack(ff_prj, -1)
        # ff_out: (B, S, vocab_size, 3)

        loss_result = []

        for i in range(self.num_features):
            loss_result.append(
                self.losses[i](
                    ff_ff_prjout[:, :, :, i].reshape(-1, self.vocab.vocab_size), 
                    target[:, :, i].reshape(-1)
                )
            )
        # loss_result: List [loss_initial, loss_rhyme, loss_tone]
        total_loss = sum(loss_result)

        return total_loss 
    
    def predict(self, src):
        pass
        # src: (B, S, 3)
        max_len = self.MAX_LENGTH
        B, S, W = src.shape 


















    def predict(self, src, max_len=None):
        pass
        device = self.config.device
        B, S, W = src.size()
        max_len = max_len if max_len is not None else self.MAX_LENGTH

        src_flat = src.view(B * S, W)

        src_mask_flat = (src_flat != self.vocab.pad_idx).unsqueeze(1).unsqueeze(2)
        
        encoder_outs_word, _ = self.word_encoder(src_flat, src_mask_flat)
        
        sent_repr = encoder_outs_word.view(B, S, W, -1).mean(dim=2)
        sent_repr = self.Sen_PE(sent_repr)
        
        sent_mask_bool = (src.sum(dim=-1) == self.vocab.pad_idx * W).to(device)
        memory = self.sentence_encoder(sent_repr, sent_mask_bool)

        ys = torch.full((B, 1), self.vocab.bos_idx, dtype=torch.long, device=device)

        for i in range(max_len):
            tgt_causal_mask = create_causal_mask(ys.size(1), device)
            tgt_padding_mask = create_padding_mask(ys, self.vocab.pad_idx).to(device)

            out = self.decoder(ys, memory, tgt_causal_mask, tgt_padding_mask, sent_mask_bool)
            
            logits = self.fc_out(out[:, -1, :])
            next_word = torch.argmax(logits, dim=-1, keepdim=True)
            
            ys = torch.cat([ys, next_word], dim=1)
            if next_word.item() == self.vocab.eos_idx:
                break

        return ys
    

    

    

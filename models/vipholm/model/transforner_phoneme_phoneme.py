import torch
from torch import nn
from vocabs.viword_vocab import Vocab
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE
from models.vipholm.utils.clone import clones
from models.vipholm.utils.padding_mask import create_padding_mask, create_standard_padding_mask
from models.vipholm.utils.causal_mask import create_causal_mask
from models.vipholm.blocks.decoder_block import TransformerDecoderBlock
from models.vipholm.blocks.encoder_block import TransformerEncoderBlock
from models.vipholm.layers.phoneme_feed_forward import FeedForward
from models.vipholm.embedding.positional_embedding import PositionalEncoding

@META_ARCHITECTURE.register()
class TransformerPhonemeLongformer(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.d_model = config.d_model
        self.MAX_LENGTH = config.inference_length
        self.config = config

        # Positional Encoding
        self.PE = PositionalEncoding(self.d_model, max_len=self.config.max_len + 10)

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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
                
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
            
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, src, trg):
        # src: (B, S, 3)
        # trg: (B, S, 3)
        B, S, W = src.shape
        _, trg_S , _ = trg.shape
        src = src[:, :self.config.max_len]
        trg = trg[:, :self.config.max_len]

        # Padding to config.max_len 
        # src (B, S, 3) with S < config.max_len
        if src.shape[1] < self.config.max_len:
            pad_length = self.config.max_len - src.shape[1]

            pad = torch.zeros(src.shape[0], pad_length, 3,\
                               device=src.device, dtype=torch.long)
            pad[:,:,0] = 3

            src = torch.cat([src, pad], dim=1)

        # trg (B, S, 3) with S < config.max_len
        if trg.shape[1] < self.config.max_len:
            pad_length = self.config.max_len - trg.shape[1]

            pad = torch.zeros(trg.shape[0], pad_length, 3, \
                               device=trg.device, dtype=torch.long)
            pad[:,:,0] = 3

            src = torch.cat([trg, pad], dim=1)

        encoder_padding_mask = create_padding_mask(src, 3)
        

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
        decoder_padding_mask = create_standard_padding_mask(decoder_input, 3)
        decoder_causal_mask = create_causal_mask(S, self.config.device)
        memory_padding_mask_bool = create_standard_padding_mask(src, 3)
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
                         decoder_padding_mask, memory_padding_mask_bool)
        
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
                    ff_prjout[:, :, :, i].reshape(-1, self.vocab.vocab_size), 
                    target[:, :, i].reshape(-1)
                )
            )
        # loss_result: List [loss_initial, loss_rhyme, loss_tone]
        total_loss = sum(loss_result)

        return 0, total_loss 
    
    def predict(self, src):
        # src: (B, S, 3)
        src = src[:, :self.config.max_len]
        B, S, W = src.shape 

        # Padding to config.max_len 
        # src (B, S, 3) with S < config.max_len
        if src.shape[1] < self.config.max_len:
            pad_length = self.config.max_len - src.shape[1]

            pad = torch.zeros(src.shape[0], pad_length, 3,\
                               device=src.device, dtype=torch.long)
            pad[:,:,0] = 3

            src = torch.cat([src, pad], dim=1)

        B = src.size(0)
        encoder_padding_mask = create_padding_mask(src, 3)
        memory_padding_mask_bool = create_standard_padding_mask(src, 3)
        # embedding
        embeds = []
        for i in range(self.num_features):
            embeds.append(self.dropout(self.src_embedding[i](src[:, :, i])))
        x = torch.cat(embeds, -1)
        x = self.linear(x)
        # x: (B, S, hidden_size)
        x = self.PE(x)
        memory = self.encoder(x, encoder_padding_mask)

        # Decoder initialize 
        # Initiate decoder's input [<BOS>, <PAD>, <PAD>]
        decoder_input = torch.empty(B, 1, self.num_features, dtype=torch.long, device=self.config.device)
        for i in range(self.num_features):
            if i == 0: 
                decoder_input[:, :, i].fill_(self.vocab.bos_idx)
            else: 
                decoder_input[:, :, i].fill_(self.vocab.pad_idx)
        # decoder_input: (batch_size, 1, 3)

        # Decoder running 
        outputs = []
        for _ in range(self.MAX_LENGTH): 
            # embedding
            embeds = []
            for i in range(self.num_features):
                embeds.append(self.dropout(self.tgt_embedding[i](decoder_input[:, :, i])))
            x = torch.cat(embeds, -1)
            x = self.linear(x)
            # x: (B, S, hidden_size)
            x = self.PE(x)

            # Masking
            trg_mask = create_standard_padding_mask(decoder_input, 3)
            trg_causal_mask = create_causal_mask(decoder_input.size(1), self.config.device)

            x = self.decoder(x, memory, trg_causal_mask, \
                         trg_mask, memory_padding_mask_bool)
            x = x[:, -1:, :]
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
            next_token = ff_prjout.argmax(dim=2)
            # next_token: (1, 1, 3)
            outputs.append(next_token)
            decoder_input = torch.cat([decoder_input, next_token], dim = 1)

            if B == 1 and next_token[:, -1, 0] == self.vocab.eos_idx:
                break
        outputs = torch.cat(outputs, dim=1) # (1, S, 3)

        return outputs
            
            
        
            
















    

    

    

import torch
from torch import nn

import math
from builders.model_builder import META_ARCHITECTURE
from models.enhanced_attn_transformer.embedding.positional_embedding import PositionalEncoding
from vocabs.hierarchy_vocab import Hierarchy_Vocab

@META_ARCHITECTURE.register()

class Hierachy_Transformer(nn.Module):
    def __init__(self, config, vocab: Hierarchy_Vocab):
        super().__init__()
    
        self.config = config    
        self.d_model = config.d_model
        self.vocab = vocab
        self.max_sentence_length = vocab.max_sentence_length + config.max_sentences * 2 # each sentence have 2 special tokens
        self.max_target_length = vocab.max_sentence_length + 2
        
        
        self.src_embedding = nn.Embedding(self.max_sentence_length, self.config.d_model)
        self.tgt_embedding = nn.Embedding(self.max_target_length, self.config.d_model)
        self.src_PE = PositionalEncoding( self.config.d_model, self.max_sentence_length)
        self.tgt_PE = PositionalEncoding(self.config.d_model, self.max_target_length)
        self.dropout = nn.Dropout(config.dropout)
        
        # Encoder 
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_head
        )
        self.encoder_word = nn.TransformerEncoder(
            self.encoder_layer, 
            self.config.n_layers
        )
        self.encoder_sent = nn.TransformerEncoder(
            self.encoder_layer, 
            self.config.n_layers
        )
        
        # Decoder 
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_head
        )
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer,
            self.config.n_layers
        )
        
        self.out = nn.Linear(config.d_model, vocab.vocab_size)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, src, tgt):
        # src: (B, S_source, W_source)
        # tgt: (B, S_target)
        
        tgt_input = tgt[:, :-1] # <bos> sentence
        tgt_output = tgt[:, 1:] # sentence <eos>
        
        # Embed 
        enc_src = self.src_embedding(src) * math.sqrt(self.config.d_model) # (B, S, W, d_model)
        enc_tgt = self.tgt_embedding(tgt_input) * math.sqrt(self.config.d_model) # (B, S_tgt, d_model)

        # Positional Encoding 
        enc_src = self.src_PE(enc_src)
        enc_tgt = self.tgt_PE(enc_tgt) 
        
        # Word encoder
        B, S, W, _  = enc_src.shape 
        
        enc_src_word = enc_src.reshape(B*S, W, -1) # (B*S, W, d_model)
            # make src mask (word)
        src_word_mask = create_padding_mask(enc_src_word.shape[1], self.vocab.pad_idx)
        
        enc_src_word_output = self.encoder_word(
            enc_src_word,
            src_word_mask
        ) # (B*S, W, d_model)
        
        # Sent encoder
            # make src mask (word)
        src_sent_mask = create_padding_mask(enc_src_word.shape[1], self.vocab.pad_idx)
        enc_src_sent_input = enc_src_word_output.reshape(B, S, W, -1)[:, :, 0, :] # (B, S, d_model)
        
        memory = self.encoder_sent(
            enc_src_sent_input,
            src_sent_mask
        ) # (B, S, d_model)
        
        # Decoder 
            # make tgt mask 
        decoder_cau_mask = create_causal_mask(tgt_input.shape[1], memory.device)
        logits = self.decoder(
            tgt_input,
            memory,
            decoder_cau_mask 
        ) # (B, S_tgt, H)
        
        out = self.out(logits) # (B, S_tgt, vocab_size)
        loss = self.loss(out.reshape(-1, self.vocab.vocab_size), tgt_output.reshape(-1))
        
        return None, loss
    
def create_causal_mask(seq_len, device):
    """
    Tạo mask Causal dạng Boolean để khớp với padding_mask.
    Logic: True = Che (Ignore), False = Nhìn (Keep).
    """
    # Tạo ma trận True ở tam giác trên (vị trí tương lai cần che)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask

def create_padding_mask(seq, pad_idx):
    """
    Tạo mask cho key_padding_mask (True là Pad).
    Shape: (Batch_Size, Seq_Len)
    """
    return (seq == pad_idx)
    
        
        
        
        
        
         

         
        

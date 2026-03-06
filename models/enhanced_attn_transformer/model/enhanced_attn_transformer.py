import torch
from torch import nn
from vocabs.hierarchy_vocab import Hierarchy_Vocab
from builders.model_builder import META_ARCHITECTURE
from models.enhanced_attn_transformer.blocks.encoder_block import TransformerEncoderBlock
from models.enhanced_attn_transformer.blocks.decoder_block import TransformerDecoderBlock
from models.enhanced_attn_transformer.blocks.standard_encoder_block import StandardTransformerEncoderBlock
from models.enhanced_attn_transformer.embedding.positional_embedding import PositionalEncoding

@META_ARCHITECTURE.register()
class EnhancedAttnTransformerModel(nn.Module):
    def __init__(self, config, vocab: Hierarchy_Vocab):
        super().__init__()
        self.vocab = vocab
        self.d_model = config.d_model
        self.MAX_LENGTH = vocab.max_sentence_length + 2
        self.config = config

        self.word_encoder = TransformerEncoderBlock(config.encoder_enhanced, vocab)
        self.sentence_encoder = StandardTransformerEncoderBlock(config.encoder_standard)
        self.decoder = TransformerDecoderBlock(config.decoder, vocab)
        
        self.Word_PE = PositionalEncoding(self.d_model, max_len=5000)
        self.Sen_PE = PositionalEncoding(self.d_model, max_len=100)
        
        self.fc_out = nn.Linear(self.d_model, vocab.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def forward(self, src, trg):
        B, S, W = src.size()
        device = src.device

        src_flat = src.view(B * S, W)
        src_mask_flat = (src_flat != self.vocab.pad_idx).unsqueeze(1).unsqueeze(2)
        
        encoder_outs_word, _ = self.word_encoder(src_flat, src_mask_flat)
        
        sent_repr = encoder_outs_word.view(B, S, W, -1).mean(dim=2)
        sent_repr = self.Sen_PE(sent_repr)
        
        sent_mask_bool = (src.sum(dim=-1) == self.vocab.pad_idx * W).to(device)
        memory = self.sentence_encoder(sent_repr, sent_mask_bool)

        trg_input = trg[:, :-1]
        
        tgt_causal_mask = create_causal_mask(trg_input.size(1), device)
        
        tgt_padding_mask = create_padding_mask(trg_input, self.vocab.pad_idx).to(device)

        decoder_outs = self.decoder(trg_input, memory, tgt_causal_mask, tgt_padding_mask, sent_mask_bool)
        
        logits = self.fc_out(decoder_outs)
        loss = self.loss(logits.view(-1, logits.size(-1)), trg[:, 1:].contiguous().view(-1))
        
        return logits, loss

    @torch.no_grad()
    def predict(self, src, max_len=None):
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
    





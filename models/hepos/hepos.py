import torch
import torch.nn as nn
from argparse import Namespace
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab
from fairseq.data.dictionary import Dictionary
from models.hepos.longbartmodel import LongBartModel 

def convert_vocab_to_fairseq_dictionary(vocab: Vocab) -> Dictionary:
    """Hàm tiện ích để chuyển đổi Vocab của bạn sang Dictionary của Fairseq."""
    d = Dictionary()
    for i in range(len(vocab)):
        d.add_symbol(vocab.get_token_from_index(i))
    d.finalize()
    return d

@META_ARCHITECTURE.register()
class HeposFairseqBaseline(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.padding_idx = vocab.pad_idx

        # --- Bước 1: Tạo đối tượng "args" giả lập cho Fairseq ---
        # Fairseq model được cấu hình bằng args, không phải object config.
        # Chúng ta sẽ chuyển đổi config của bạn thành args.
        args = self.create_fairseq_args()

        # --- Bước 2: Tạo Dictionary của Fairseq từ Vocab của bạn ---
        self.fairseq_dictionary = convert_vocab_to_fairseq_dictionary(vocab)

        # --- Bước 3: Khởi tạo mô hình LongBartModel của tác giả ---
        # Lớp này sẽ sử dụng các file sinkhorn.py, sparse_decoder.py, v.v.
        self.model = LongBartModel.build_model(args, None)
        
        # Đồng bộ embedding layer nếu cần
        # Fairseq model có embed_tokens riêng, chúng ta cần đảm bảo nó có đúng kích thước
        self.model.encoder.embed_tokens = self.build_embedding(
            len(self.fairseq_dictionary), config.d_model, self.padding_idx
        )
        self.model.decoder.embed_tokens = self.model.encoder.embed_tokens

    def create_fairseq_args(self) -> Namespace:
        """Tạo một đối tượng Namespace chứa các cấu hình cần thiết cho LongBartModel."""
        args = Namespace(
            # Cấu hình kiến trúc cơ bản
            encoder_embed_dim=self.config.d_model,
            encoder_ffn_embed_dim=self.config.d_model * 4,
            encoder_layers=self.config.num_encoder_layers,
            encoder_attention_heads=self.config.num_heads,
            decoder_embed_dim=self.config.d_model,
            decoder_ffn_embed_dim=self.config.d_model * 4,
            decoder_layers=self.config.num_decoder_layers,
            decoder_attention_heads=self.config.num_heads,
            activation_fn="gelu",
            dropout=self.config.dropout,
            attention_dropout=self.config.attention_dropout,
            max_source_positions=self.config.max_src_len,
            max_target_positions=self.config.max_tgt_len,
            no_token_positional_embeddings=False,
            encoder_learned_pos=True,
            layernorm_embedding=True,
            share_decoder_input_output_embed=True,
            
            # --- Cấu hình CỐT LÕI cho baseline ---
            # Kích hoạt Sinkhorn cho Encoder
            sinkhorn=True, 
            bucket_size=self.config.sinkhorn_bucket_size,
            encoder_not_hybrid=True, # Sử dụng sinkhorn cho tất cả các lớp

            # Kích hoạt HEPOS (stride) cho Decoder
            decoder_divide_attention=True,
            divide_type='stride',
            divide_ratio=self.config.hepos_stride_size,
            
            # Tắt các tùy chọn không cần thiết khác
            lsh_attention=False,
            sw=False,
            encoder_linear=False,
            decoder_linear_attention=False,
            # (Thêm các cờ khác và đặt là False nếu cần)
        )
        return args

    def build_embedding(self, num_embeddings, embedding_dim, padding_idx):
        """Hàm tiện ích để tạo embedding layer."""
        m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
        return m

    def forward(self, src_tokens, tgt_tokens, **kwargs):
        """
        Hàm forward của wrapper.
        Nó nhận input từ pipeline của bạn và chuyển đổi cho phù hợp với Fairseq model.
        """
        # Fairseq decoder cần `prev_output_tokens` (target được dịch phải 1 vị trí)
        # để dự đoán token tiếp theo (teacher forcing).
        prev_output_tokens = self.prepare_decoder_input(tgt_tokens)

        # Gọi hàm forward của mô hình Fairseq gốc
        # `src_lengths` có thể không cần thiết nếu bạn đã padding tất cả các câu
        # về cùng một độ dài trong batch.
        decoder_output, extra = self.model(
            src_tokens=src_tokens,
            src_lengths=None, 
            prev_output_tokens=prev_output_tokens
        )
        
        # decoder_output chính là logits bạn cần cho hàm loss.
        return decoder_output

    def prepare_decoder_input(self, tgt_tokens):
        """Tạo prev_output_tokens từ target tokens cho teacher forcing."""
        # Dịch phải target sequence
        prev_output_tokens = tgt_tokens.clone()
        # Chèn token bắt đầu câu (EOS trong Fairseq thường là 2) ở đầu
        prev_output_tokens[:, 0] = self.fairseq_dictionary.eos()
        prev_output_tokens[:, 1:] = tgt_tokens[:, :-1]
        return prev_output_tokens
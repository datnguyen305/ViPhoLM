from fairseq.models import register_model, register_model_architecture
from fairseq.models.bart.model import BARTModel, bart_large_architecture
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules import TransformerEncoderLayer, LayerDropModuleList, PositionalEmbedding
import torch.nn as nn
from models.hepos.sinkhorn_attn import SinkhornSelfAttention
from .sparse_decoder import SparseTransformerDecoder # Decoder này sẽ được chúng ta chỉnh sửa ở Bước 2

@register_model('baseline_longbart')
class LongBartModel(BARTModel):
    @staticmethod
    def add_args(parser):
        """Chỉ giữ lại các đối số cần thiết cho baseline."""
        super(LongBartModel, LongBartModel).add_args(parser)
        parser.add_argument('--bucket_size', default=64, type=int, help='Bucket size for Sinkhorn attention')
        parser.add_argument('--divide_ratio', default=4, type=int, help='Divide ratio for HEPOS (stride size)')
        
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        """Luôn xây dựng Encoder với Sinkhorn attention."""
        return SinkhornEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        """Luôn xây dựng Decoder sẽ sử dụng HEPOS."""
        # Đảm bảo cờ cần thiết được bật để sparse_decoder.py chọn đúng attention
        args.decoder_divide_attention = True
        args.divide_type = 'stride' # Đây chính là logic của HEPOS
        return SparseTransformerDecoder(
            args, tgt_dict, embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


@register_model_architecture('baseline_longbart', 'baseline_longbart_large')
def longbart_architecture(args):
    """Định nghĩa kiến trúc large cho baseline."""
    args.max_source_positions = getattr(args, 'max_source_positions', 10240)
    bart_large_architecture(args)

# LỚP ENCODER ĐÃ ĐƯỢC TINH GỌN
class SinkhornEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        
        # Luôn xây dựng các lớp encoder với Sinkhorn
        self.layers = nn.ModuleList(
            [SinkhornTransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        
        # Giữ lại logic Positional Embedding cho văn bản dài
        self.embed_positions_new = PositionalEmbedding(
            args.max_source_positions, self.embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        )
        # (Các phần khởi tạo khác có thể giữ lại)

# LỚP ENCODER LAYER ĐÃ ĐƯỢC TINH GỌN
class SinkhornTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)
        # Ghi đè self-attention để luôn là Sinkhorn
        self.self_attn = self.build_sinkhorn_self_attention(self.embed_dim, args)

    def build_sinkhorn_self_attention(self, embed_dim, args):
        return SinkhornSelfAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            bucket_size=args.bucket_size
        )
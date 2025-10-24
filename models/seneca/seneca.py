import torch
from torch import nn
from builders.model_builder import META_ARCHITECTURE 
from models.seneca.extractor import EntityAwareExtractor
from models.seneca.generator import AbstractGenerator
from vocabs.vocab import Vocab 

@META_ARCHITECTURE.register()
class SENECAModel(nn.Module):
    """
    Kiến trúc SENECA hoàn chỉnh, kết hợp Extractor và Generator.
    Đây là model sẽ được đăng ký và sử dụng làm baseline trong framework META.
    """
    def __init__(self, cfg):
        """
        Hàm khởi tạo, nhận vào một đối tượng config (cfg) chung.
        cfg nên chứa các cấu hình con cho Extractor và Generator.
        """
        super().__init__()
        self.cfg = cfg
        self.vocab = Vocab(cfg.MODEL.VOCAB)

        vocab_size = self.vocab.vocab_size
        
        # --- Khởi tạo Module 1: Extractor ---
        # Lấy config dành riêng cho Extractor
        ext_cfg = cfg.MODEL.SENECA.EXTRACTOR
        self.extractor = EntityAwareExtractor(
            vocab_size=cfg.MODEL.VOCAB_SIZE,
            emb_dim=ext_cfg.EMB_DIM,
            conv_hidden=ext_cfg.CONV_HIDDEN,
            lstm_hidden=ext_cfg.LSTM_HIDDEN,
            lstm_layer=ext_cfg.LSTM_LAYER,
            bidirectional=ext_cfg.BIDIRECTIONAL,
            n_hop=ext_cfg.N_HOP,
            dropout=ext_cfg.DROPOUT
        )
        
        # --- Khởi tạo Module 2: Generator ---
        # Lấy config dành riêng cho Generator
        gen_cfg = cfg.MODEL.SENECA.GENERATOR
        self.generator = AbstractGenerator(
            vocab_size=cfg.MODEL.VOCAB_SIZE,
            emb_dim=gen_cfg.EMB_DIM,
            n_hidden=gen_cfg.N_HIDDEN,
            bidirectional=gen_cfg.BIDIRECTIONAL,
            n_layer=gen_cfg.N_LAYER,
            dropout=gen_cfg.DROPOUT
        )

    def _get_sents_from_indices(self, article_sents, indices):
        """
        Hàm tiện ích để lấy nội dung các câu từ chỉ số mà Extractor trả về.
        - article_sents: [B, max_sents, max_words]
        - indices: List của list, ví dụ [[1, 5, 2], [0, 2]]
        """
        # Đây là một cách triển khai đơn giản, bạn có thể cần tối ưu hóa
        batch_extracted_sents = []
        for i, idx_list in enumerate(indices):
            # Sắp xếp các chỉ số để giữ trật tự câu gốc
            sorted_indices = sorted(idx_list)
            sents = [article_sents[i][j] for j in sorted_indices]
            batch_extracted_sents.append(torch.stack(sents))
        
        # Cần padding lại batch này để có cùng số câu
        # (Logic padding có thể được đặt trong một hàm utils chung)
        padded_batch = nn.utils.rnn.pad_sequence(batch_extracted_sents, batch_first=True, padding_value=0)
        return padded_batch

    def forward(self, batch, mode='train_supervised'):
        """
        Forward pass chính của mô hình.
        - batch: Một dict chứa tất cả dữ liệu cần thiết.
        - mode: Chế độ hoạt động ('train_supervised', 'inference').
        """
        if mode == 'train_supervised':
            # --- Giai đoạn huấn luyện có giám sát ---
            # Mục tiêu là huấn luyện riêng lẻ Extractor và Generator
            
            # 1. Huấn luyện Extractor
            # Extractor học cách dự đoán các chỉ số câu đúng (target_indices)
            extraction_scores = self.extractor(
                batch['article_sents'], batch['sent_nums'],
                batch['clusters'], batch['cluster_nums'],
                batch['target_indices']  # Dùng teacher forcing
            )
            
            # 2. Huấn luyện Generator
            # Generator học cách tóm tắt từ các câu trích xuất đúng (ground-truth)
            # Lấy các câu trích xuất đúng từ dữ liệu
            gt_extracted_sents = self._get_sents_from_indices(batch['article_sents'], batch['target_indices'])
            
            summary_logits = self.generator(
                article=gt_extracted_sents,
                art_lens=batch['gt_extracted_len'], # Cần được cung cấp trong batch
                abstract=batch['abstract'],
                extend_art=batch['extend_art'], # Dữ liệu cho cơ chế copy
                extend_vsize=batch['extend_vsize']
            )
            
            # Trả về output để tính 2 loss riêng biệt
            return {
                'extraction_scores': extraction_scores,
                'summary_logits': summary_logits
            }
            
        elif mode == 'inference':
            # --- Giai đoạn suy luận (tạo tóm tắt thực tế) ---
            # Pipeline end-to-end
            
            # 1. Extractor trích xuất ra các chỉ số câu tốt nhất
            # Hàm `extract` trong extractor cần được triển khai để không dùng teacher forcing
            extracted_indices = self.extractor.extract(
                batch['article_sents'], batch['sent_nums'],
                batch['clusters'], batch['cluster_nums'], k=4 # k là số câu muốn trích xuất
            )
            
            # 2. Lấy nội dung các câu đã trích xuất
            extracted_sents = self._get_sents_from_indices(batch['article_sents'], extracted_indices)
            # 3. Generator tạo tóm tắt từ các câu đã trích xuất
            summaries = self.generator.batched_beamsearch(
                article=extracted_sents,
                art_lens=[len(s) for s in extracted_sents],
                extend_art=batch['extend_art'],
                extend_vsize=batch['extend_vsize'],
                # Lấy token ID trực tiếp từ đối tượng vocab của model
                go=self.vocab.bos_idx, 
                eos=self.vocab.eos_idx, 
                unk=self.vocab.unk_idx,
                max_len=self.cfg.INFERENCE.MAX_LEN, 
                beam_size=self.cfg.INFERENCE.BEAM_SIZE
            )
            
            return summaries
        else:
            raise ValueError(f"Chế độ không hợp lệ: {mode}")
        
    def decode(self, tensor_indices):
        """Sử dụng vocab nội bộ để giải mã một tensor chỉ số thành câu."""
        return self.vocab.decode_sentence(tensor_indices)
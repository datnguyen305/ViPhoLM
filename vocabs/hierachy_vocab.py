import torch
import json
import math
from collections import Counter
from tqdm import tqdm
from typing import List

from builders.vocab_builder import META_VOCAB
from .utils import preprocess_sentence
<<<<<<< HEAD
from .vocab import Vocab
@META_VOCAB.register()
class Hierachy_Vocab(Vocab):
    def __init__(self, config):
        # 1. Khởi tạo lớp cha
=======
from vocab import Vocab

@META_VOCAB.register()
class Hierachy_Vocab(Vocab):
    def __init__(self, config):
        # 1. Khởi tạo các thuộc tính cơ bản
>>>>>>> models/fast
        super().__init__(config) 
        self.initialize_pos_ner_stoi()
        self.idf_dict = {}

<<<<<<< HEAD
        # 2. SỬA LỖI Ở ĐÂY: Truy cập cụ thể vào 'train' thay vì toàn bộ 'path'
        if hasattr(config, "path"):
            # Kiểm tra xem config.path là chuỗi hay là node cấu hình
            if isinstance(config.path, str):
                train_path = config.path
            else:
                # Nếu là node (CfgNode), lấy đường dẫn file train
                train_path = config.path.train
            
            self._auto_build_idf(train_path)

    def initialize_pos_ner_stoi(self):
        # Danh sách nhãn POS underthesea
=======
        # 2. Tự động chạy build IDF nếu có đường dẫn dữ liệu trong config
        # Thông thường config.path chứa đường dẫn file json (train.json)
        if hasattr(config, "path"):
            self._auto_build_idf(config.path)

    def _auto_build_idf(self, data_path: str):
        """Đọc file data và chuẩn bị danh sách văn bản để tính IDF"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            all_docs = []
            for key in data:
                doc_sents = []
                # Lấy tất cả câu trong phần source của mỗi item
                raw_source = data[key].get("source", {})
                for _, sents in raw_source.items():
                    doc_sents.extend(sents)
                all_docs.append(doc_sents)
            
            # Chạy hàm tính toán
            self.build_idf_dict(all_docs)
        except Exception as e:
            print(f"Lỗi khi tự động build IDF: {e}")

    def build_idf_dict(self, all_documents: List[List[str]]):
        num_docs = len(all_documents)
        doc_count = Counter()

        print(f"--- Đang tính toán IDF cho {num_docs} văn bản ---")
        for doc in tqdm(all_documents):
            unique_words = set()
            for sent in doc:
                # Dùng hàm preprocess đã import
                tokens = preprocess_sentence(sent)
                unique_words.update(tokens)
            
            for word in unique_words:
                doc_count[word] += 1

        # Tính toán giá trị IDF cuối cùng
        for word, count in doc_count.items():
            # Thêm +1 để tránh log(0) và smoothing
            self.idf_dict[word] = math.log(num_docs / (count + 1))
        
        print(f"Đã tính xong IDF cho {len(self.idf_dict)} từ.")

    def initialize_pos_ner_stoi(self):
        # Nhãn POS Tiếng Việt (Mục 2.2)
>>>>>>> models/fast
        vn_pos_list = [
            'N', 'V', 'A', 'P', 'R', 'L', 'M', 'E', 'C', 'I', 'T', 'Y', 
            'Np', 'Nc', 'Nu', 'Ny', 'X', 'CH', 'B', 'S', 'Vb'
        ]
        self.pos_stoi = {v: k + 1 for k, v in enumerate(vn_pos_list)}
        self.pos_stoi["<pad>"] = 0
        self.pos_stoi["unk"] = len(self.pos_stoi)

<<<<<<< HEAD
        # Danh sách nhãn NER underthesea
=======
        # Nhãn NER Tiếng Việt (Mục 2.2)
>>>>>>> models/fast
        vn_ner_list = [
            'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 
            'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O'
        ]
        self.ner_stoi = {v: k + 1 for k, v in enumerate(vn_ner_list)}
        self.ner_stoi["<pad>"] = 0
        self.ner_stoi["unk"] = len(self.ner_stoi)

<<<<<<< HEAD
    def _auto_build_idf(self, data_path: str):
        """Tự động đọc JSON và tính IDF"""
        print(f"Đang tính IDF từ file: {data_path}")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            all_docs = []
            for key in data:
                doc_sents = []
                # Hỗ trợ cả cấu trúc "source": "câu" hoặc "source": {"0": "câu"}
                raw_source = data[key].get("source", {})
                
                if isinstance(raw_source, str):
                    doc_sents.append(raw_source)
                elif isinstance(raw_source, dict):
                    for _, sents in raw_source.items():
                        if isinstance(sents, list):
                            doc_sents.extend(sents)
                        else:
                            doc_sents.append(str(sents))
                
                all_docs.append(doc_sents)
            
            self.build_idf_dict(all_docs)
            print(f"Hoàn thành tính IDF. Tổng số từ trong từ điển IDF: {len(self.idf_dict)}")
            
        except Exception as e:
            print(f"Lỗi build IDF: {e}")

    def build_idf_dict(self, all_documents: List[List[str]]):
        num_docs = len(all_documents)
        doc_count = Counter()
        
        # Dùng tqdm để hiển thị tiến độ nếu file lớn
        for doc in tqdm(all_documents, desc="Tính IDF"):
            unique_words = set()
            for sent in doc:
                # preprocess_sentence cần trả về list các token (str)
                tokens = preprocess_sentence(sent)
                unique_words.update(tokens)
            
            for word in unique_words:
                doc_count[word] += 1
                
        for word, count in doc_count.items():
            self.idf_dict[word] = math.log(num_docs / (count + 1))

=======
>>>>>>> models/fast
    @property 
    def pos_size(self) -> int:
        return len(self.pos_stoi)
    
    @property
    def ner_size(self) -> int:
        return len(self.ner_stoi)
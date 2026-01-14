from torch.utils.data import Dataset
import json
import torch
# Thay spacy bằng underthesea
from underthesea import pos_tag, ner

from builders.dataset_builder import META_DATASET
from utils.instance import Instance
from vocabs.hierachy_vocab import Hierachy_Vocab
from vocabs.utils import preprocess_sentence
@META_DATASET.register()
class TextSumDataset_Hierarchical(Dataset):
    def __init__(self, config, vocab: Hierachy_Vocab) -> None:
        super().__init__()

        path: str = config.path
        self._data = json.load(open(path, encoding='utf-8'))
        self._keys = list(self._data.keys())
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._data)
    
    def _get_tfidf_bin(self, word, all_tokens):
        # Tính TF đơn giản
        tf = all_tokens.count(word) / max(len(all_tokens), 1)
        # IDF lấy từ từ điển đã tính sẵn trong vocab
        idf = self._vocab.idf_dict.get(word, 1.0)
        tfidf = tf * idf
        # Chia thành 10 bins (0-9)
        return int(min(tfidf * 100, 9))

    def __getitem__(self, index: int) -> Instance:
        key = self._keys[index]
        item = self._data[key]
        
        raw_source = item["source"]
        all_sentences = []
        for _, sents in raw_source.items():
            all_sentences.extend(sents)

        # Lấy toàn bộ token để tính TF-IDF
        full_text_tokens = [t.lower() for s in all_sentences for t in preprocess_sentence(s)]

        encoded_source_list = []
        for sent in all_sentences:
            # 1. Tiền xử lý lấy tokens chuẩn của bạn
            tokens = preprocess_sentence(sent) 
            
            # 2. Trích xuất POS và NER bằng underthesea
            # pos_tag trả về list of tuples: [('Từ', 'Loại'), ...]
            # ner trả về list of tuples: [('Từ', 'Loại', 'Thực thể'), ...]
            try:
                ut_pos = pos_tag(sent)
                ut_ner = ner(sent)
                
                # Ánh xạ nhãn sang ID (lấy index 1 cho POS và index 2 cho NER)
                # Dùng [:len(tokens)] để đảm bảo độ dài khớp với tokens đã preprocess
                pos_tags = [self._vocab.pos_stoi.get(t[1], 0) for t in ut_pos][:len(tokens)]
                ner_tags = [self._vocab.ner_stoi.get(t[2], 0) for t in ut_ner][:len(tokens)]
            except:
                # Trường hợp lỗi thư viện hoặc câu rỗng
                pos_tags = [0] * len(tokens)
                ner_tags = [0] * len(tokens)

            # Padding nếu tags bị ngắn hơn tokens do khác biệt bộ tách từ
            if len(pos_tags) < len(tokens):
                pos_tags += [0] * (len(tokens) - len(pos_tags))
            if len(ner_tags) < len(tokens):
                ner_tags += [0] * (len(tokens) - len(ner_tags))

            # 3. Tính TF-IDF bins
            tfidf_bins = [self._get_tfidf_bin(t, full_text_tokens) for t in tokens]

            # 4. Mã hóa ID từ vựng
            word_ids = [self._vocab.stoi.get(t, self._vocab.unk_idx) for t in tokens]

            # Lưu vào cấu trúc phân cấp
            encoded_source_list.append({
                'word_ids': torch.LongTensor(word_ids),
                'pos_ids': torch.LongTensor(pos_tags),
                'ner_ids': torch.LongTensor(ner_tags),
                'tfidf_ids': torch.LongTensor(tfidf_bins)
            })
        
        # Mã hóa Target
        encoded_target = self._vocab.encode_sentence(item["target"])

        return Instance(
            id = key,
            input_features = encoded_source_list, 
            label = encoded_target
        )
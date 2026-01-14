from torch.utils.data import Dataset
import json
import torch
from underthesea import pos_tag, ner
from builders.dataset_builder import META_DATASET
from utils.instance import Instance
from vocabs.utils import preprocess_sentence
# Đảm bảo import đúng vocab
# from vocabs.hierachy_vocab import Hierachy_Vocab 

@META_DATASET.register()
class TextSumDataset_Hierarchical(Dataset):
    def __init__(self, config, vocab) -> None:
        super().__init__()
        path: str = config.path
        self._data = json.load(open(path, encoding='utf-8'))
        self._keys = list(self._data.keys())
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._data)
    
    def _get_tfidf_bin(self, word, all_tokens):
        tf = all_tokens.count(word) / max(len(all_tokens), 1)
        idf = self._vocab.idf_dict.get(word, 1.0)
        tfidf = tf * idf
        return int(min(tfidf * 100, 9))

    def __getitem__(self, index: int) -> Instance:
        key = self._keys[index]
        item = self._data[key]
        
        raw_source = item["source"]
        all_sentences = []
        # Xử lý input dạng dict hoặc list
        if isinstance(raw_source, dict):
            for _, sents in raw_source.items():
                all_sentences.extend(sents)
        elif isinstance(raw_source, list):
            all_sentences = raw_source
        else:
            all_sentences = [str(raw_source)]

        # 1. Thu thập OOV list
        oov_list = []
        full_text_tokens = []
        processed_sentences_tokens = []
        
        for s in all_sentences:
            tokens = preprocess_sentence(s)
            processed_sentences_tokens.append(tokens)
            full_text_tokens.extend([t.lower() for t in tokens])
            
            for t in tokens:
                if t not in self._vocab.stoi and t not in oov_list:
                    oov_list.append(t)

        # 2. Xử lý Source Features
        encoded_source_list = []
        vocab_size = self._vocab.vocab_size

        for i, sent in enumerate(all_sentences):
            tokens = processed_sentences_tokens[i]
            
            # POS & NER
            try:
                ut_pos = pos_tag(sent)
                ut_ner = ner(sent)
                # Map tags (giả sử vocab đã có pos_stoi/ner_stoi)
                pos_tags = [self._vocab.pos_stoi.get(t[1], 0) for t in ut_pos][:len(tokens)]
                ner_tags = [self._vocab.ner_stoi.get(t[2], 0) for t in ut_ner][:len(tokens)]
            except:
                pos_tags, ner_tags = [0] * len(tokens), [0] * len(tokens)

            # Padding tags nếu thiếu
            if len(pos_tags) < len(tokens): pos_tags += [0] * (len(tokens) - len(pos_tags))
            if len(ner_tags) < len(tokens): ner_tags += [0] * (len(tokens) - len(ner_tags))

            tfidf_bins = [self._get_tfidf_bin(t, full_text_tokens) for t in tokens]

            word_ids = [self._vocab.stoi.get(t, self._vocab.unk_idx) for t in tokens]
            
            # Extended IDs cho Pointer
            extended_word_ids = []
            for t in tokens:
                if t in self._vocab.stoi:
                    extended_word_ids.append(self._vocab.stoi[t])
                else:
                    extended_word_ids.append(vocab_size + oov_list.index(t))

            encoded_source_list.append({
                'word_ids': word_ids,
                'extended_word_ids': extended_word_ids, 
                'pos_ids': pos_tags,
                'ner_ids': ner_tags,
                'tfidf_ids': tfidf_bins,
                'tokens': tokens # Lưu tokens gốc để debug nếu cần
            })
        
        # 3. Xử lý Target & Shifted Label (QUAN TRỌNG)
        target_tokens = preprocess_sentence(item["target"])
        
        # 3a. Target Input (Shifted Right) cho Decoder: <BOS> ... <EOS> (dùng ID thường)
        # Lưu ý: Input decoder thường bắt đầu bằng BOS và không cần EOS
        dec_input_tokens = [self._vocab.bos_token] + target_tokens
        word_ids_input = [self._vocab.stoi.get(t, self._vocab.unk_idx) for t in dec_input_tokens]

        # 3b. Target Label cho Loss: ... <EOS> (dùng ID mở rộng)
        # Label là từ tiếp theo cần dự đoán, kết thúc bằng EOS
        label_tokens = target_tokens + [self._vocab.eos_token]
        extended_label_ids = []
        for t in label_tokens:
            if t in self._vocab.stoi:
                extended_label_ids.append(self._vocab.stoi[t])
            elif t in oov_list:
                extended_label_ids.append(vocab_size + oov_list.index(t))
            else:
                extended_label_ids.append(self._vocab.unk_idx)

        # TRẢ VỀ INSTANCE VỚI ĐẦY ĐỦ TRƯỜNG
        return Instance(
            id = key,
            input_features = encoded_source_list, 
            shifted_right_label = word_ids_input,  # <--- Trường này đang thiếu ở code cũ
            label = extended_label_ids,            # Target cho Loss
            oov_list = oov_list,
            vocab_size = vocab_size
        )
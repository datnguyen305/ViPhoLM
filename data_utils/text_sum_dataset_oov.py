from torch.utils.data import Dataset
import json
from vocabs.utils import preprocess_sentence
from builders.dataset_builder import META_DATASET
from utils.instance import Instance
from vocabs.vocab import Vocab
import torch

@META_DATASET.register()
class TextSumDatasetOOV(Dataset):
    def __init__(self, config, vocab: Vocab) -> None:
        super().__init__()

        path: str = config.path
        self._data = json.load(open(path,  encoding='utf-8'))
        self._keys = list(self._data.keys())
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Instance:
        key = self._keys[index]
        item = self._data[key]
        
        paragraphs = item["source"]
        paragraphs = [" ".join(paragraph) for _, paragraph in paragraphs.items()]
        source_text = "<nl>".join(paragraphs) # new line mark
        target_text = item["target"]

        # --- BẮT ĐẦU LOGIC PGN MỚI ---
        
        # 1. Tokenize văn bản
        source_tokens = preprocess_sentence(source_text)
        target_tokens = preprocess_sentence(target_text)

        # 2. Tạo danh sách OOV (từ điển tạm thời) cho câu nguồn
        oov_list: list[str] = []
        encoded_source = [self._vocab.bos_idx] # Bắt đầu với <bos>
        
        for token in source_tokens:
            if token in self._vocab.stoi:
                # Từ này có trong từ vựng
                encoded_source.append(self._vocab.stoi[token])
            else:
                # Từ này là OOV
                if token not in oov_list:
                    oov_list.append(token)
                
                # Index OOV = vocab_size + vị trí của nó trong list OOV
                oov_index = self._vocab.vocab_size + oov_list.index(token)
                encoded_source.append(oov_index)
                
        encoded_source.append(self._vocab.eos_idx) # Kết thúc với <eos>

        # 3. Encode câu target (nhãn), trỏ tới OOV nếu có thể
        encoded_target = [self._vocab.bos_idx]
        
        for token in target_tokens:
            if token in self._vocab.stoi:
                # Từ này có trong từ vựng
                encoded_target.append(self._vocab.stoi[token])
            else:
                # Từ này là OOV
                if token in oov_list:
                    # Nếu từ OOV này đã xuất hiện trong NGUỒN
                    # -> trỏ tới nó
                    oov_index = self._vocab.vocab_size + oov_list.index(token)
                    encoded_target.append(oov_index)
                else:
                    # Nếu là OOV nhưng không có trong nguồn
                    # -> coi như <unk>
                    encoded_target.append(self._vocab.unk_idx)
                    
        encoded_target.append(self._vocab.eos_idx)
        
        # Chuyển sang Tensor
        encoded_source = torch.tensor(encoded_source, dtype=torch.long)
        encoded_target = torch.tensor(encoded_target, dtype=torch.long)
        
        # --- KẾT THÚC LOGIC PGN MỚI ---

        shifted_right_label = encoded_target[1:]
        
        # Trả về Instance, thêm trường 'oov_list'
        return Instance(
            id = key,
            input_ids = encoded_source,
            label = encoded_target,
            shifted_right_label = shifted_right_label,
            oov_list = oov_list  # <-- RẤT QUAN TRỌNG
        )
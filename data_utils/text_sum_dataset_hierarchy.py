from torch.utils.data import Dataset
import json
import torch 
from builders.dataset_builder import META_DATASET
from utils.instance import Instance
from vocabs.hierarchy_vocab import Hierarchy_Vocab

"""
    Ready
"""

@META_DATASET.register()
class TextSumDatasetHierarchy(Dataset):
    def __init__(self, config, vocab: Hierarchy_Vocab) -> None:
        super().__init__()

        path: str = config.path
        self._data = json.load(open(path,  encoding='utf-8'))
        self._keys = list(self._data.keys())
        self._vocab = vocab
        self.MAX_SENTENCE_LENGTH = config.get("max_sentence_length", 50) 
        self.MAX_SENTS = config.get("max_sentences", 10)
        self.pad_idx = self._vocab.pad_idx
    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Instance:
        key = self._keys[index]
        item = self._data[key]

        paragraphs = item["source"]
        document = [s for paragraph in paragraphs.values() for s in paragraph]

        # 1. Mã hóa tài liệu
        # Giả sử encode_document trả về list các Tensor 1D
        sentence_tensors = self._vocab.encode_document(document)

        # 2. Làm phẳng (Flatten) để collate_fn có thể thực hiện cat và pad
        # Chúng ta nối các câu lại thành 1 chuỗi ID phẳng duy nhất
        flat_input_ids = torch.cat(sentence_tensors) if len(sentence_tensors) > 0 else torch.tensor([], dtype=torch.long)

        target = item["target"]
        encoded_target = self._vocab.encode_sentence(target)

        # TRẢ VỀ TÊN KHỚP VỚI COLLATE_FN
        return Instance(
            id=key,
            input_ids=flat_input_ids,          # Dùng cho logic cat/pad trong collate
            encoded_document=flat_input_ids,   # Thêm key này để tránh lỗi AttributeError
            label=encoded_target,
            shifted_right_label=encoded_target[1:],
            pad_idx=self.pad_idx
        )

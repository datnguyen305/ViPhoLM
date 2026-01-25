from torch.utils.data import Dataset
import json
from typing import List

from builders.dataset_builder import META_DATASET
from utils.instance import Instance
from vocabs.viword_vocab import ViWordVocab
from vocabs.utils import preprocess_sentence # Đừng quên import cái này

@META_DATASET.register()
class TextSumDatasetPhoneme(Dataset):
    def __init__(self, config, vocab: ViWordVocab) -> None:
        super().__init__()

        self.path: str = config.path
        self._data = json.load(open(self.path, encoding='utf-8'))
        self._keys = list(self._data.keys())
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Instance:
        key = self._keys[index]
        item = self._data[key]
        
        # --- 1. Xử lý Source ---
        raw_source = item["source"]
        paragraphs = [" ".join(paragraph) for _, paragraph in raw_source.items()]
        source_str = " <nl> ".join(paragraphs) 
    
        source_tokens = preprocess_sentence(source_str)
        encoded_source = self._vocab.encode_caption(source_tokens)

        # --- 2. Xử lý Target ---
        target_str = item["target"]
        target_tokens = preprocess_sentence(target_str)
        
        # encoded_target sẽ có dạng: [BOS, w1, w2, ..., EOS]
        encoded_target = self._vocab.encode_caption(target_tokens)

        shifted_right_label = encoded_target[1:]
       
        return Instance(
            id = key,
            input_ids = encoded_source,     
            label = encoded_target, 
            shifted_right_label = shifted_right_label                        
        )
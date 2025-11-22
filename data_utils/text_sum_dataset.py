from torch.utils.data import Dataset
import json

from builders.dataset_builder import META_DATASET
from utils.instance import Instance
from vocabs.vocab import Vocab

@META_DATASET.register()
class TextSumDataset(Dataset):
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
        source = "<nl>".join(paragraphs) # new line mark
        target = item["target"]

        encoded_source = self._vocab.encode_sentence(source)
        encoded_target = self._vocab.encode_sentence(target)

        shifted_right_label = encoded_target[1:]
       
        return Instance(
            id = key,
            input_ids = encoded_source,
            label = encoded_target,
            shifted_right_label = shifted_right_label
        )

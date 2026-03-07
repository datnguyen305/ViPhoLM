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
        self.MAX_SENTENCE_LENGTH = config.get("max_sentence_length", 40) 
        self.MAX_SENTS = config.get("max_sentences", 40)
        self.pad_idx = self._vocab.pad_idx
    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Instance:
        key = self._keys[index]
        item = self._data[key]

        paragraphs = item["source"]
        document = [s for paragraph in paragraphs.values() for s in paragraph]

        sentence_tensors = self._vocab.encode_document(document, max_doc_len = self.MAX_SENTS, max_sent_len = self.MAX_SENTENCE_LENGTH)

        target = item["target"]
        encoded_target = self._vocab.encode_sentence(target)

        return Instance(
            id=key,
            input_ids=sentence_tensors,
            label=encoded_target,
            shifted_right_label=encoded_target[1:],
        )

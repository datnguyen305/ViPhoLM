from torch.utils.data import Dataset
import json
from typing import List
import torch
from builders.dataset_builder import META_DATASET
from utils.instance import Instance
from vocabs.viword_vocab import Vocab
from vocabs.utils import preprocess_sentence
from torch.nn.utils.rnn import pad_sequence


@META_DATASET.register()
class TextSumDatasetSeneca(Dataset):
    """
    Dataset cho SENECA
    Trả về:
      - sentences: List[List[int]]      (N sentences)
      - entities:  List[List[int]]      (E entity clusters)
      - summary:   List[int]
    """

    def __init__(self, config, vocab: Vocab) -> None:
        super().__init__()

        self.path: str = config.path
        self._data = json.load(open(self.path, encoding="utf-8"))
        self._keys = list(self._data.keys())
        self._vocab = vocab

        self.max_sent = getattr(config, "max_sent", 30)
        self.max_len = getattr(config, "max_len", 50)

    def __len__(self) -> int:
        return len(self._keys)

    def encode_sentence(self, sent: str) -> torch.Tensor:
        return self._vocab.encode_sentence(sent)[: self.max_len]



    def __getitem__(self, index: int) -> Instance:
        key = self._keys[index]
        item = self._data[key]

        # -------- 1. SOURCE → sentence-level --------
        raw_source = item["source"]
        sentences: List[List[int]] = []

        for _, paragraph in raw_source.items():
            for sent in paragraph:
                sentences.append(self.encode_sentence(sent))
                if len(sentences) >= self.max_sent:
                    break
            if len(sentences) >= self.max_sent:
                break

        # -------- 2. ENTITY CLUSTERS --------
        entity_clusters: List[List[int]] = []

        for cluster in item.get("entities", []):
            entity_text = " ".join(cluster)
            entity_clusters.append(
                self._vocab.encode_sentence(entity_text)
            )


        # -------- 3. TARGET SUMMARY --------
        encoded_target = self._vocab.encode_sentence(item["target"])
        shifted_right_label = encoded_target[1:]

        return Instance(
            id=key,
            sentences=sentences,                 # List[List[int]]
            entities=entity_clusters,             # List[List[int]]
            label=encoded_target,                 # summary
            shifted_right_label=shifted_right_label
        )


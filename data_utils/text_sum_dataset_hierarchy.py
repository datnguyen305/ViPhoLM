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

        # document aka sentence
        document = [s + "<nl>" for paragraph in paragraphs.values() for s in paragraph]

        sentences = [self._vocab.encode_sentence(sentence) for sentence in document]
        # sentences = [max_sentences, max_sentence_length]

        input_ids = torch.full((self.MAX_SENTS, self.MAX_SENTENCE_LENGTH), self.pad_idx, dtype=torch.long)
        
        for i, s_tokens in enumerate(sentences[:self.MAX_SENTS]):
            valid_tokens = s_tokens[:self.MAX_SENTENCE_LENGTH]
            input_ids[i, :len(valid_tokens)] = valid_tokens.clone()


        target = item["target"]
        encoded_target = self._vocab.encode_sentence(target)
        shifted_right_label = encoded_target[1:]
       
        return Instance(
            id = key,
            input_ids = input_ids,
            label = encoded_target,
            shifted_right_label = shifted_right_label,
        )

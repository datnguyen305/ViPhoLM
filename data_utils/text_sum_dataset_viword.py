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
class TextSumDatasetViword(Dataset):
    def __init__(self, config, vocab: Hierarchy_Vocab) -> None:
        super().__init__()
        path: str = config.path
        self._data = json.load(open(path, encoding='utf-8'))
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
        # Flatten paragraphs into a list of sentences
        document = [s for paragraph in paragraphs.values() for s in paragraph]
        
        # Encode document - returns list of tensors
        input_ids = self._vocab.encode_document(document)
        
        # Limit number of sentences
        if len(input_ids) > self.MAX_SENTS:
            input_ids = input_ids[:self.MAX_SENTS]
        
        # Truncate each sentence to max length
        truncated_input_ids = []
        for sent_tensor in input_ids:
            if sent_tensor.size(0) > self.MAX_SENTENCE_LENGTH:
                sent_tensor = sent_tensor[:self.MAX_SENTENCE_LENGTH]
            truncated_input_ids.append(sent_tensor)
        
        target = item["target"]
        encoded_target = self._vocab.encode_sentence(target)
        
        return Instance(
            id=key,
            input_ids=truncated_input_ids,  # list[Tensor], each is (Si,)
            label=encoded_target,            # (T,)
            shifted_right_label=encoded_target,  # Keep full for collate_fn
            pad_idx=self.pad_idx
        )
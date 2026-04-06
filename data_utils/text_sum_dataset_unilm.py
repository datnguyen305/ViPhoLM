from torch.utils.data import Dataset
import json
import torch
from builders.dataset_builder import META_DATASET
from utils.instance import Instance
from vocabs.vocab import Vocab

@META_DATASET.register()
class TextSumDatasetUniLM(Dataset):
    def __init__(self, config, vocab: Vocab) -> None:
        super().__init__()

        path: str = config.path
        self._data = json.load(open(path, encoding='utf-8'))
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

        dict_source = {"source": source}
        dict_target = {"target": target}

        """SOURCE"""
        encoded_source = self._vocab.encode_sentence(dict_source["source"], next(iter(dict_source)))
        # encoded_source: <bos> sentence <eos>
        # type: Tensor, shape: (S_source)
        
        encoded_source_type = torch.ones(encoded_source.shape[0], \
            device=encoded_source.device, dtype = torch.long)
        # encoded_source_type: (S_source)
        
        src_len = encoded_source.shape[0]
        
        """TARGET"""
        encoded_target = self._vocab.encode_sentence(dict_target["target"], next(iter(dict_target)))
        # encoded_target: sentence <eos>
        # type: Tensor, shape: (S_target)
        
        encoded_target_type = torch.zeros(encoded_target.shape[0], \
            device=encoded_target.device, dtype = torch.long)
        # encoded_target_type:(S_target) 
        
        tgt_len = encoded_target.shape[0]
        
        """COMBINE"""
        input_ids = torch.cat((encoded_source, encoded_target), dim=0)
        # input_ids: (S_source + S_target)
        
        input_type_ids = torch.cat((encoded_source_type, encoded_target_type), dim=0)
        # input_type_ids: (S_source + S_target)

        return Instance(
            id = key,
            input_ids = input_ids,
            input_type_ids = input_type_ids,
            src_len = torch.tensor(src_len),
            tgt_len = torch.tensor(tgt_len)
        )

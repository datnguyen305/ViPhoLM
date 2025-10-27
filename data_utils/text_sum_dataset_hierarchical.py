from torch.utils.data import Dataset
import json
import torch
from torch.nn import functional as F
from builders.dataset_builder import META_DATASET
from utils.instance import Instance
from vocabs.vocab import Vocab

@META_DATASET.register()
class HierarchicalTextSumDataset(Dataset):
    def __init__(self, config, vocab: Vocab) -> None:
        super().__init__()
        self._data = json.load(open(config.path, encoding='utf-8'))
        self._keys = list(self._data.keys())
        self._vocab = vocab
        self.max_sents = config.max_sents
        self.max_sent_len = config.max_sent_len

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Instance:
        key = self._keys[index]
        item = self._data[key]
        
        # Process source paragraphs into sentences
        paragraphs = item["source"]
        sentences = []
        for _, para in paragraphs.items():
            # Split paragraph into sentences
            sents = [sent.strip() for sent in " ".join(para).split(".") if sent.strip()]
            sentences.extend(sents)
        
        # Encode sentences
        encoded_sents = []
        for sent in sentences[:self.max_sents]:
            tokens = self._vocab.encode_sentence(sent)
            # Pad/truncate sentence
            tokens = tokens[:self.max_sent_len]
            tokens = F.pad(tokens, (0, self.max_sent_len - len(tokens)), value=self._vocab.pad_idx)
            encoded_sents.append(tokens)
        
        # Pad number of sentences if needed
        while len(encoded_sents) < self.max_sents:
            pad_sent = torch.full((self.max_sent_len,), self._vocab.pad_idx)
            encoded_sents.append(pad_sent)
            
        # Stack into tensor [S_s, S_w]
        source = torch.stack(encoded_sents)
        
        # Encode target (remains flat)
        target = self._vocab.encode_sentence(item["target"])
        shifted_right_label = target[1:]

        return Instance(
            id = key,
            input_ids = source,        # [S_s, S_w]
            label = target,            # [T]
            shifted_right_label = shifted_right_label  # [T-1]
        )
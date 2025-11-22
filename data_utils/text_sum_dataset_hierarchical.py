from torch.utils.data import Dataset
import torch
import json
from builders.dataset_builder import META_DATASET
from utils.instance import Instance
from vocabs.vocab import Vocab
from .text_sum_dataset import TextSumDataset
from vocabs.hierarchicalVocab import HierarchicalVocab

@META_DATASET.register()
class TextSumDatasetHierarchical(TextSumDataset):
    def __init__(self, config, vocab: HierarchicalVocab, max_sentences=50, max_words=50) -> None:
        super().__init__(config, vocab)
        self.max_sentences = max_sentences  # S
        self.max_words = max_words          # W

    def __getitem__(self, index: int) -> Instance:
        key = self._keys[index]
        item = self._data[key]

        # Lấy danh sách câu
        paragraphs = item["source"]
        paragraphs = [" ".join(paragraph) for _, paragraph in paragraphs.items()]
        target = item["target"]

        # Encode các câu
        encoded_source = self._vocab.encode_sentences(paragraphs[:self.max_sentences], self.max_words)  # (S, W)
        sentence_lengths = [min(len(s.split()), self.max_words) for s in paragraphs[:self.max_sentences]]

        # Nếu số câu < max_sentences, pad thêm câu toàn PAD
        num_pad_sents = self.max_sentences - encoded_source.size(0)
        if num_pad_sents > 0:
            pad_sentence = torch.full((self.max_words,), self._vocab.pad_idx, dtype=torch.long)
            encoded_source = torch.cat([encoded_source, pad_sentence.unsqueeze(0).repeat(num_pad_sents, 1)], dim=0)
            sentence_lengths.extend([0] * num_pad_sents)

        # Encode target
        encoded_target = self._vocab.encode_sentence(target)
        if encoded_target.dim() == 0:
            encoded_target = encoded_target.unsqueeze(0)  # Chuyển thành tensor 1 chiều

        shifted_right_label = encoded_target[1:]

        return Instance(
            id=key,
            input_ids=encoded_source,          # (S, W)
            input_lengths=torch.tensor(sentence_lengths),  # số từ thực tế mỗi câu
            label=encoded_target,
            shifted_right_label=shifted_right_label
        )
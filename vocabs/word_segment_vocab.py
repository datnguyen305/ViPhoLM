import json
import torch
from collections import Counter
from typing import List
from builders.vocab_builder import META_VOCAB
from vocabs.vocab import Vocab
from underthesea import word_tokenize

def segment_sentence(sentence: str) -> List[str]:
    """
    Word segmentation cho tiếng Việt.
    Output dạng list token, ví dụ:
    'Trí tuệ nhân tạo' -> ['Trí_tuệ', 'nhân_tạo']
    """
    if not sentence or not sentence.strip():
        return []
    return word_tokenize(sentence, format="text").split()

@META_VOCAB.register()
class WordSegmentVocab(Vocab):
    """
    Word-segmentation-based vocabulary cho Text Summarization.
    - Token level: word (đã segment)
    - Encode: 1D Tensor
    - Decode: string
    """

    def __init__(self, config):
        self.initialize_special_tokens(config)
        self.make_vocab(config)

    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token

        self.specials = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]

        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        
    def make_vocab(self, config):
        json_dirs = [
            config.path.train,
            config.path.dev,
            config.path.test,
        ]

        counter = Counter()
        self.max_sentence_length = 0

        for json_dir in json_dirs:
            with open(json_dir, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key in data:
                item = data[key]

                # -------- SOURCE --------
                paragraphs = item["source"]
                paragraphs = [
                    " ".join(paragraph)
                    for _, paragraph in paragraphs.items()
                ]
                source_text = " ".join(paragraphs)
                source_tokens = segment_sentence(source_text)
                counter.update(source_tokens)

                # -------- TARGET --------
                target_text = item["target"]
                target_tokens = segment_sentence(target_text)
                counter.update(target_tokens)

                self.max_sentence_length = max(
                    self.max_sentence_length,
                    len(target_tokens),
                )

        min_freq = max(config.min_freq, 1)

        # sort theo freq giảm dần, sau đó alphabet
        words_and_freqs = sorted(
            counter.items(),
            key=lambda x: (-x[1], x[0])
        )

        vocab_tokens = []
        for word, freq in words_and_freqs:
            if freq < min_freq:
                break
            vocab_tokens.append(word)

        itos = self.specials + vocab_tokens

        self.itos = {i: tok for i, tok in enumerate(itos)}
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        
    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """
        Encode sentence thành tensor chỉ số:
        [<bos>, w1, w2, ..., <eos>]
        """
        tokens = segment_sentence(sentence)

        vec = (
            [self.bos_idx]
            + [self.stoi.get(tok, self.unk_idx) for tok in tokens]
            + [self.eos_idx]
        )

        return torch.tensor(vec).long()
    
    def decode_sentence(
        self,
        sentence_vecs: torch.Tensor,
        join_words: bool = True
    ) -> List[str]:
        """
        sentence_vecs: (batch_size, max_len)
        """
        sentences = []

        for vec in sentence_vecs:
            tokens = [
                self.itos[idx]
                for idx in vec.tolist()
                if self.itos[idx] not in self.specials
            ]

            sent = " ".join(tokens)
            sentences.append(sent)

        return sentences
    
    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
    
    @property
    def max_len(self) -> int:
        return self.max_sentence_length + 2  # +2 for <bos> and <eos>
    
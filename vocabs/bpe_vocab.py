import torch
import sentencepiece as spm
from builders.vocab_builder import META_VOCAB
from typing import List
import json
import os

@META_VOCAB.register()
class BPE_Vocab(object):

    def __init__(self, config):
        self.initialize_special_tokens(config)
        self.make_vocab(config)

    def initialize_special_tokens(self, config):

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

    def dataset_iterator(self, json_dirs):
        self.max_sentence_length = 0

        for json_dir in json_dirs:

            data = json.load(open(json_dir, encoding="utf-8"))

            for key in data:

                item = data[key]

                paragraphs = item["source"]
                paragraphs = [
                    " ".join(paragraph)
                    for _, paragraph in paragraphs.items()
                ]

                source = " ".join(paragraphs)
                target = item["target"]
                yield source
                yield target

    def make_vocab(self, config):
        
        self.max_sentence_length = 0

        json_dirs = [
            config.path.train,
            config.path.dev,
            config.path.test,
        ]

        model_dir = config.model_prefix
        os.makedirs(model_dir, exist_ok=True)

        model_file = config.model_prefix + ".model"

        if not os.path.exists(model_file):
            spm.SentencePieceTrainer.train(
                sentence_iterator=self.dataset_iterator(json_dirs),
                model_prefix=config.model_prefix,
                vocab_size=config.vocab_size,
                model_type="bpe",
                character_coverage=1.0,
                pad_id=self.pad_idx,
                bos_id=self.bos_idx,
                eos_id=self.eos_idx,
                unk_id=self.unk_idx,
            )

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)

        self._vocab_size = self.sp.get_piece_size()

        self.itos = {
            i: self.sp.id_to_piece(i)
            for i in range(self._vocab_size)
        }

        self.stoi = {
            v: k for k, v in self.itos.items()
        }

        for json_dir in json_dirs:

            data = json.load(open(json_dir, encoding="utf-8"))

            for key in data:

                item = data[key]

                paragraphs = item["source"]
                paragraphs = [
                    " ".join(paragraph)
                    for _, paragraph in paragraphs.items()
                ]

                source = " ".join(paragraphs)
                target = item["target"]

                source_ids = self.sp.encode(source)
                target_ids = self.sp.encode(target)

                if self.max_sentence_length < len(target_ids):
                    self.max_sentence_length = len(target_ids)

    @property
    def vocab_size(self):
        return self._vocab_size

    def encode_sentence(self, sentence: str) -> torch.Tensor:

        ids = self.sp.encode(sentence, out_type=int)

        ids = [self.bos_idx] + ids + [self.eos_idx]

        return torch.tensor(ids).long()

    def decode_sentence(
        self,
        sentence_vecs: torch.Tensor,
        join_words=True,
    ) -> List[str]:

        sentences = []

        for vec in sentence_vecs:

            ids = vec.tolist()

            sentence = self.sp.decode(ids)

            if join_words:
                sentences.append(sentence)
            else:
                sentences.append(sentence.split())

        return sentences

    def __len__(self):
        return self._vocab_size
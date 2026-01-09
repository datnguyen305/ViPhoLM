import os
import json
import torch
from typing import List

from builders.vocab_builder import META_VOCAB
from vocabs.vocab import Vocab
from vocabs.utils import preprocess_sentence
from vocabs.Vietnamese_utils import analyze_Vietnamese, compose_word
from typing import *

@META_VOCAB.register()
class ViWordVocab(Vocab):
    def __init__(self, config):
        self.tokenizer = config.TOKENIZER

        self.initialize_special_tokens(config)
        
        phonemes = self.make_vocab(config)
        phonemes = list(phonemes)
        self.itos = {
            i: tok for i, tok in enumerate(self.specials + phonemes)
        }

        self.stoi = {
            tok: i for i, tok in enumerate(self.specials + phonemes)
        }

        # only padding token is not allowed to be shown
        self.specials = [self.padding_token]

    def initialize_special_tokens(self, config) -> None:
        self.padding_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token
        
        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]

        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def make_vocab(self, config):
        # Lấy list đường dẫn từ config (Đã sửa ở bước trước)
        json_paths = [config.TRAIN, config.DEV, config.TEST]
        phonemes = set()

        # Collect token stats from each JSON
        for path in json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON path not found: {path}")
            
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key in data:
                item = data[key]
                
                # --- SỬA ĐOẠN NÀY ---
                
                # 1. Xử lý SOURCE (Văn bản gốc)
                # Dữ liệu source là Dict -> Cần nối lại thành String
                raw_source = item["source"]
                if isinstance(raw_source, dict):
                    # Logic giống hệt trong ViTextSumDataset
                    paragraphs = [" ".join(p) for _, p in raw_source.items()]
                    source_text = " ".join(paragraphs)
                else:
                    # Phòng trường hợp nó đã là string
                    source_text = str(raw_source)

                # 2. Xử lý TARGET (Tóm tắt) - QUAN TRỌNG
                # Bạn cũng cần từ vựng của phần tóm tắt để Decoder học
                target_text = item.get("target", "")
                
                # 3. Gộp cả hai để quét từ vựng
                full_text = source_text + " " + target_text
                
                # --------------------

                words = preprocess_sentence(full_text)
                
                for word in words:
                    components = analyze_Vietnamese(word)
                    if components:
                        phonemes.update([phoneme for phoneme in components if phoneme])

        return phonemes

    def encode_caption(self, caption: List[str]) -> torch.Tensor:
        syllables = [
            (self.bos_idx, self.pad_idx, self.pad_idx, self.pad_idx)
        ]
        for word in caption:
            components = analyze_Vietnamese(word)
            if components:
                syllables.append([
                    self.stoi[phoneme] if phoneme else self.pad_idx for phoneme in components
                ])
            else:
                syllables.append(
                    (self.unk_idx, self.pad_idx, self.pad_idx, self.pad_idx)
                )

        syllables.append(
            (self.eos_idx, self.pad_idx, self.pad_idx, self.pad_idx)
        )

        vec = torch.tensor(syllables).long()

        return vec

    def decode_caption(self, caption_vec: torch.Tensor, join_words=True):
        assert caption_vec.dim() == 2
        syllable_ids = caption_vec.tolist()
        syllables = [
            (
                self.itos[phoneme_idx] for phoneme_idx in phoneme_ids
            ) for phoneme_ids in syllable_ids
        ]
        sentence = []
        for phonemes in syllables:
            onset, medial, nucleus, coda = phonemes
            # turn phoneme into None if they are in special tokens
            onset = None if onset in self.specials else onset
            medial = None if medial in self.specials else medial
            nucleus = None if nucleus in self.specials else nucleus
            coda = None if coda in self.specials else coda
            word = compose_word(onset, medial, nucleus, coda)
            if word:
                sentence.append(word)
            else:
                if onset == self.bos_token:
                    sentence.append(self.bos_token)
                    continue
                
                if onset == self.eos_token:
                    sentence.append(self.eos_token)
                    continue
                
                sentence.append(self.unk_token)

        # remove the <bos> and <eos> token
        if sentence[0] == self.bos_token:
            sentence = sentence[1:]
        if sentence[-1] == self.eos_token:
            sentence = sentence[:-1]

        if join_words:
            return " ".join(sentence)
        else:
            return sentence

    def decode_batch_caption(self, caption_batch: torch.Tensor, join_words=True):
        assert caption_batch.dim() == 3
        captions = [
            self.decode_caption(caption_vec, join_words) for caption_vec in caption_batch
        ]

        return captions
    
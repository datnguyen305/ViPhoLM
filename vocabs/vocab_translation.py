import os
import json
import torch
from typing import List
from builders.vocab_builder import META_VOCAB
from vocabs.viword_vocab import Vocab
from vocabs.utils import preprocess_sentence
from vocabs.Vietnamese_utils import analyse_Vietnamese, compose_word
from typing import *

@META_VOCAB.register()
class MTVocab(Vocab):
    def __init__(self, config):
        self.tokenizer = config.TOKENIZER
        self.initialize_special_tokens(config) # general config

        # vietnamese vocab
        vietnamese_phonemes = self.make_vocab_vietnamese(config.vietnamese) # vietnamese config
        vietnamese_phonemes = list(vietnamese_phonemes)
        self.vietnamese_itos = {
            i: tok for i, tok in enumerate(self.specials + vietnamese_phonemes)
        }

        self.vietnamese_stoi = {
            tok: i for i, tok in enumerate(self.specials + vietnamese_phonemes)
        }

        # english vocab initiate
        self.make_vocab_english(config.english) # english config


    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token
        
        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    # vietnamese methods 
    @property
    def vietnamese_vocab_size(self) -> int:
        return len(self.vietnamese_stoi)

    def make_vocab_vietnamese(self, config):
        # Lấy list đường dẫn từ config (Đã sửa ở bước trước)
        json_paths = [config.path.train, config.path.dev, config.path.test]
        phonemes = set()
        self.vietnamese_max_sentence_length = 0
        # Collect token stats from each JSON
        for path in json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON path not found: {path}")
            
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for sample in data:
                # sample['vietnamese']
                # sample['english']
                
                raw_source = sample["vietnamese"]
                vietnamese_text = raw_source

                vietnamese_words = preprocess_sentence(vietnamese_text)
                
                for word in vietnamese_words:
                    components = analyse_Vietnamese(word)
                    if components:
                        phonemes.update([phoneme for phoneme in components if phoneme])

                if self.vietnamese_max_sentence_length < len(vietnamese_words):
                    self.vietnamese_max_sentence_length = len(vietnamese_words)

        return phonemes

    def encode_caption_vietnamese(self, caption: List[str]) -> torch.Tensor:
        syllables = [
            (self.bos_idx, self.pad_idx, self.pad_idx)
        ]
        for word in caption:
            components = analyse_Vietnamese(word)
            if components:
                syllables.append([
                    self.vietnamese_stoi[phoneme] if phoneme else self.pad_idx for phoneme in components
                ])
            else:
                syllables.append(
                    (self.unk_idx, self.pad_idx, self.pad_idx)
                )

        syllables.append(
            (self.eos_idx, self.pad_idx, self.pad_idx)
        )

        vec = torch.tensor(syllables).long()

        return vec

    def decode_caption_vietnamese(self, caption_vec: torch.Tensor, join_words=True):
        assert caption_vec.dim() == 2
        syllable_ids = caption_vec.tolist()
        
        syllables = [
            [self.vietnamese_itos[idx] for idx in phoneme_ids]
            for phoneme_ids in syllable_ids
        ]
        
        sentence = []
        for phonemes in syllables:
            initial, rhyme, tone = phonemes

            # Check initial có phải là special_token(bos, eos) không
            if initial in self.specials:
                if initial == self.bos_token:
                    sentence.append(self.bos_token)
                elif initial == self.eos_token:
                    sentence.append(self.eos_token)
                continue
            
            # Check phonemes phù hợp cho hàm compose_word
            clean_initial = initial
            clean_rhyme = '' if rhyme in self.specials else rhyme
            clean_tone = '-' if tone in self.specials else tone
            
            try:
                word = compose_word(clean_initial, clean_rhyme, clean_tone)
                if word:
                    sentence.append(word)
                else:
                    sentence.append(self.unk_token)
            except Exception as e:
                sentence.append(self.unk_token)

        # Bỏ bos_token, eos_token
        if len(sentence) > 0:
            if sentence[0] == self.bos_token:
                sentence = sentence[1:]
        if len(sentence) > 0:
            if sentence[-1] == self.eos_token:
                sentence = sentence[:-1]

        # Bỏ qua các unk_token
        sentence = [word for word in sentence if word != self.unk_token]

        if join_words:
            return " ".join(sentence)
        else:
            return sentence

    def decode_batch_caption_vietnamese(self, caption_batch: torch.Tensor, join_words=True):
        assert caption_batch.dim() == 3
        captions = [
            self.decode_caption_vietnamese(caption_vec, join_words) for caption_vec in caption_batch
        ]

        return captions
    
    # english methods
    def make_vocab_english(self, config):
            json_dirs = [config.path.train, config.path.dev, config.path.test]
            counter = Counter()
            self.english_max_sentence_length = 0

            for json_dir in json_dirs:
                data = json.load(open(json_dir, encoding='utf-8'))

                for sample in data:
                    # sample['vietnamese']
                    # sample['english']

                    english_source = sample["english"]

                    english_sent = preprocess_sentence(english_source)
                    counter.update(english_sent)
                    if self.english_max_sentence_length < len(english_sent):
                        self.english_max_sentence_length = len(english_sent)

            min_freq = max(config.min_freq, 1)

            # sort by frequency, then alphabetically
            words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
            words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
            itos = []
            for word, freq in words_and_frequencies:
                if freq < min_freq:
                    break
                itos.append(word)
            itos = self.specials + itos

            self.english_itos = {i: tok for i, tok in enumerate(itos)}
            self.english_stoi = {tok: i for i, tok in enumerate(itos)}

    @property
    def english_vocab_size(self) -> int:
        return len(self.english_stoi)
    
    def encode_sentence_english(self, sentence: str) -> torch.Tensor:
        """ Turn a sentence into a vector of indices and a sentence length """
        sentence = preprocess_sentence(sentence)
        vec = [self.bos_idx] + [self.english_stoi[token] if token in self.english_stoi else self.unk_idx for token in sentence] + [self.eos_idx]
        vec = torch.Tensor(vec).long()

        return vec
    
    def decode_sentence_english(self, sentence_vecs: torch.Tensor, join_words=True) -> List[str]:
        '''
            sentence_vecs: (bs, max_length)
        '''
        sentences = []
        for vec in sentence_vecs:
            question = " ".join([self.english_itos[idx] for idx in vec.tolist() if self.english_itos[idx] not in self.specials])
            if join_words:
                sentences.append(question)
            else:
                sentences.append(question.strip().split())

        return sentences

    


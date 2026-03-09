import os
import json
import torch
from typing import List

from builders.vocab_builder import META_VOCAB
from vocabs.vocab import Vocab
from vocabs.utils import preprocess_sentence
from vocabs.Vietnamese_utils import analyse_Vietnamese, compose_word
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
    
    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def make_vocab(self, config):
        # Lấy list đường dẫn từ config (Đã sửa ở bước trước)
        json_paths = [config.TRAIN, config.DEV, config.TEST]
        phonemes = set()
        self.max_sentence_length = 0
        # Collect token stats from each JSON
        for path in json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON path not found: {path}")
            
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key in data:
                item = data[key]
                
                
                raw_source = item["source"]
                if isinstance(raw_source, dict):
                    paragraphs = [" ".join(p) for _, p in raw_source.items()]
                    source_text = " ".join(paragraphs)
                else:
                    source_text = str(raw_source)

                target_text = item.get("target", "")
                
                full_text = source_text + " " + target_text
                

                words = preprocess_sentence(full_text)
                
                for word in words:
                    components = analyse_Vietnamese(word)
                    if components:
                        phonemes.update([phoneme for phoneme in components if phoneme])

                if self.max_sentence_length < len(target_text):
                    self.max_sentence_length = len(target_text)

        return phonemes

    def encode_caption(self, caption: List[str]) -> torch.Tensor:
        syllables = [
            (self.bos_idx, self.pad_idx, self.pad_idx)
        ]
        for word in caption:
            components = analyse_Vietnamese(word)
            if components:
                syllables.append([
                    self.stoi[phoneme] if phoneme else self.pad_idx for phoneme in components
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

    def decode_caption(self, caption_vec: torch.Tensor, join_words=True):
        assert caption_vec.dim() == 2
        syllable_ids = caption_vec.tolist()
        
        # Sử dụng .get() để tránh lỗi KeyError nếu model dự đoán index lạ
        syllables = [
            [self.itos.get(idx, self.unk_token) for idx in phoneme_ids]
            for phoneme_ids in syllable_ids
        ]
        
        sentence = []
        for phonemes in syllables:
            # Giải nén 3 thành phần
            initial, rhyme, tone = phonemes
            
            # 1. Kiểm tra Token đặc biệt (Thoát sớm để an toàn)
            if initial in self.specials:
                if initial == self.bos_token:
                    sentence.append(self.bos_token)
                elif initial == self.eos_token:
                    sentence.append(self.eos_token)
                # Bỏ qua <pad> và các token khác, không gọi compose_word
                continue

            # 2. Chuẩn bị dữ liệu cho compose_word
            # 'initial' giữ nguyên (hoặc None nếu bạn muốn khớp chính xác dict)
            # 'rhyme' không được là None để tránh lỗi .startswith() trong utils
            # 'tone' phải là '-' nếu model dự đoán ra token đặc biệt ở vị trí tone
            
            clean_initial = initial
            clean_rhyme = '' if rhyme in self.specials else rhyme
            clean_tone = '-' if tone in self.specials else tone
            
            try:
                # Gọi hàm ghép từ
                word = compose_word(clean_initial, clean_rhyme, clean_tone)
                
                if word:
                    sentence.append(word)
                else:
                    # Nếu compose_word trả về None hoặc rỗng (do sai luật ngữ pháp)
                    sentence.append(self.unk_token)
            except Exception as e:
                # Phòng hờ trường hợp utils vẫn crash do logic bên trong
                sentence.append(self.unk_token)

        # 3. Hậu xử lý: Xóa <bos> và <eos> ở đầu/cuối câu
        if len(sentence) > 0:
            if sentence[0] == self.bos_token:
                sentence = sentence[1:]
        if len(sentence) > 0:
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
    
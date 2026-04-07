import torch
from gensim.models import FastText
from builders.vocab_builder import META_VOCAB
from .utils import preprocess_sentence
from collections import Counter
from typing import List
import json
import os

@META_VOCAB.register()
class UniLM_Vocab(object):
    def __init__(self, config):
        self.initialize_special_token(config)
        self.make_vocab(config)
        self.fasttext_dim = config.get("fasttext_dim", 300)
        if config.get("train_fasttext", False):
            self.train_fasttext(config)
    
    def initialize_special_token(self, config):
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token 
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token 
        
        self.specials = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token
        ]
        
        self.pad_idx = 0 
        self.bos_idx = 1 
        self.eos_idx = 2 
        self.unk_idx = 3 
        
    def make_vocab(self, config):
        self.max_input_length = 0
        self.max_sentence_length = 0
        self.sentences_for_fasttext = []
        counter = Counter()
        json_dirs = [
            config.path.train,
            config.path.dev,
            config.path.test
        ]
        
        for json_dir in json_dirs: 
            data = json.load(open(json_dir, encoding='utf-8'))
            for key in data: 
                item = data[key]
                paragraphs = item["source"]
                paragraphs = [" ".join(paragraph) for _, paragraph in paragraphs.items()]
                source = "<nl>".join(paragraphs) # new line mark
                # source exp: "Nếu cơn đau trở nên nặng hơn mỗi khi"
                
                fragmented_source = preprocess_sentence(source)
                # fragmented_source exp: ["Nếu", "cơn", "đau", "trở", "nên", "nặng"]
                counter.update(fragmented_source)
                self.sentences_for_fasttext.append(fragmented_source)
                
                target = item["target"]
                fragmented_target = preprocess_sentence(target)
                counter.update(fragmented_target)
                self.sentences_for_fasttext.append(fragmented_target)
                
                total_input_len = len(fragmented_source) + len(fragmented_target)
                
                if self.max_sentence_length < len(fragmented_target):
                    self.max_sentence_length = len(fragmented_target)
                if self.max_sentence_length < total_input_len:
                    self.max_input_length = total_input_len
                
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

        self.itos = {i: tok for i, tok in enumerate(itos)}
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        
    def train_fasttext(self, config):
        ft_model = FastText(
            sentences=self.sentences_for_fasttext, 
            vector_size=self.fasttext_dim, 
            window=config.window_size, 
            min_count=config.min_freq, 
            workers=config.num_workers
        )
        
        vocab_size = len(self.itos)
        self.embedding_matrix = torch.zeros((vocab_size, self.fasttext_dim))
        # embedding_matrix: (vocab_size, 300) but values == 0

        # Initial specials tokens with random values
        for i in range(1, len(self.specials)): 
             self.embedding_matrix[i] = torch.randn(self.fasttext_dim)

        # Mapping text 
        for i in range(len(self.specials), vocab_size):
            word = self.itos[i]
            if word in ft_model.wv:
                # Lấy vector từ FastText
                self.embedding_matrix[i] = torch.tensor(ft_model.wv[word].copy())
            else:
                # Nếu từ bị lọt (hiếm khi xảy ra), khởi tạo ngẫu nhiên
                self.embedding_matrix[i] = torch.randn(self.fasttext_dim)    
        
    @property 
    def vocab_size(self) -> int:
        return len(self.itos)
    
    def encode_sentence(self, sentence: str, type: str) -> torch.Tensor:
        """ 
        Turn a sentence into a vector of indices and a sentence length
        Returns: 
            source: <bos> sentence_idx <eos>
            target: sentence_idx <eos>
        """
        sentence = preprocess_sentence(sentence)
        
        if type == "source":
            vec = [self.bos_idx] + [self.stoi[token] if token in self.stoi else self.unk_idx for token in sentence] + [self.eos_idx]
            vec = torch.Tensor(vec).long()
        else: 
            vec = [self.stoi[token] if token in self.stoi else self.unk_idx for token in sentence] + [self.eos_idx]
            vec = torch.Tensor(vec).long()
            
        return vec
    
    def decode_sentence(self, sentence_vecs: torch.Tensor, join_words=True) -> List[str]:
        '''
            sentence_vecs: (bs, max_length)
        '''
        sentences = []
        for vec in sentence_vecs:
            question = " ".join([self.itos[idx] for idx in vec.tolist() if self.itos[idx] not in self.specials])
            if join_words:
                sentences.append(question)
            else:
                sentences.append(question.strip().split())

        return sentences

    def __len__(self):
        return len(self.itos)
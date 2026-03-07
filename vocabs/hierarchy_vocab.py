import torch
import json
from collections import Counter
from builders.vocab_builder import META_VOCAB
from typing import List
from .utils import preprocess_sentence
from vocabs.vocab import Vocab

@META_VOCAB.register()
class Hierarchy_Vocab(Vocab):
    def __init__(self, config):
        self.initialize_special_tokens(config)
        self.make_vocab(config)
    
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
    
    """
        Ready
    """
    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        counter = Counter()
        self.max_sentence_length = 0
        
        for json_dir in json_dirs:
            data = json.load(open(json_dir, encoding='utf-8'))
            for key in data:
                item = data[key]
                
                paragraphs = item["source"]
                paragraphs = [" ".join(paragraph) for _, paragraph in paragraphs.items()]
                source = "<nl>".join(paragraphs)  # new line mark
                
                paragraphs = preprocess_sentence(source)
                counter.update(paragraphs)
                
                target = item["target"]
                target = preprocess_sentence(target)
                counter.update(target)
                
                if self.max_sentence_length < len(target):
                    self.max_sentence_length = len(target)
        
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
    
    """
        Ready
    """
    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
    
    """
        Ready - hopefully
    """
    def encode_document(self, document: List[str], max_doc_len = 40, max_sent_len = 40) -> List[torch.Tensor]:
        """
        document: list of sentences (strings)
        return: A single 2D Tensor [max_doc_len, max_sent_len]
        """
        sentences = []
        for i, sent in enumerate(document):
            if i >= max_doc_len: 
                break
                
            sent_vec = self.encode_sentence(sent) 
            if len(sent_vec) < max_sent_len:
                sent_vec += [self.pad_idx] * (max_sent_len - len(sent_vec))
            else:
                sent_vec = sent_vec[:max_sent_len]
                
            sentences.append(sent_vec)

        while len(sentences) < max_doc_len:
            empty_sentence = [self.pad_idx] * max_sent_len
            sentences.append(empty_sentence)

        return torch.tensor(sentences, dtype=torch.long)
    
    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """Turn a sentence into a vector of indices"""
        sentence = preprocess_sentence(sentence)
        vec = [self.bos_idx] + [
            self.stoi.get(token, self.unk_idx) for token in sentence
        ] + [self.eos_idx]
        
        return vec
    
    def decode_sentence(self, sentence_vecs: torch.Tensor, join_words=True) -> List[str]:
        '''
            sentence_vecs: (bs, max_length)
        '''
        sentences = []
        for vec in sentence_vecs:
            question = " ".join([
                self.itos[idx] for idx in vec.tolist() 
                if idx in self.itos and self.itos[idx] not in self.specials
            ])
            if join_words:
                sentences.append(question)
            else:
                sentences.append(question.strip().split())
        return sentences
    
    def __eq__(self, other: "Vocab"):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True
    
    def __len__(self):
        return len(self.itos)
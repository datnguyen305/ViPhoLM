import torch
import spacy
from collections import defaultdict
from typing import List, Tuple
from .vocab import Vocab
from .utils import preprocess_sentence
from builders.vocab_builder import META_VOCAB
import json

@META_VOCAB.register()
class RichVocab(Vocab):
    def __init__(self, config):
        super().__init__(config)
        # Load spaCy model for Vietnamese
        self.nlp = spacy.load(config.spacy_model)
        
        # Initialize POS and NER mappings
        self.pos_tags = defaultdict(int)
        self.ner_tags = defaultdict(int)
        
        # Compute TF-IDF and create tag mappings
        self.compute_tfidf(config)
        self.create_tag_mappings(config)

    def compute_tfidf(self, config):
        """Compute TF and IDF scores for vocabulary"""
        self.tf = defaultdict(float)
        self.idf = defaultdict(float)
        total_docs = 0
        
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        for json_dir in json_dirs:
            data = json.load(open(json_dir, encoding='utf-8'))
            total_docs += len(data)
            
            for item in data.values():
                # Process document
                doc_words = preprocess_sentence(item["target"])
                word_count = len(doc_words)
                
                # Count word frequencies
                word_freqs = defaultdict(int)
                for word in doc_words:
                    word_freqs[word] += 1
                
                # Update TF and document frequencies
                for word, count in word_freqs.items():
                    self.tf[word] += count / word_count
                    self.idf[word] += 1
        
        # Compute IDF scores
        for word in self.idf:
            doc_freq = self.idf[word]
            # Convert to tensor and compute IDF
            idf_score = torch.tensor(total_docs / (1 + doc_freq), dtype=torch.float)
            self.idf[word] = torch.log(idf_score)
            
    def create_tag_mappings(self, config):
        """Create mappings for POS and NER tags"""
        json_dirs = [config.path.train]  # Only use training data for tag collection
        
        for json_dir in json_dirs:
            data = json.load(open(json_dir, encoding='utf-8'))
            for item in data.values():
                doc = self.nlp(item["target"])
                
                # Collect POS tags
                for token in doc:
                    self.pos_tags[token.pos_] += 1
                
                # Collect NER tags
                for ent in doc.ents:
                    self.ner_tags[ent.label_] += 1
        
        # Create mappings
        self.pos_stoi = {tag: idx for idx, tag in enumerate(self.pos_tags.keys(), start=4)}  # Start after special tokens
        self.pos_itos = {idx: tag for tag, idx in self.pos_stoi.items()}
        
        self.ner_stoi = {tag: idx for idx, tag in enumerate(self.ner_tags.keys(), start=4)}
        self.ner_itos = {idx: tag for tag, idx in self.ner_stoi.items()}

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """
        Encode a sentence into token indices and rich features
        
        Returns:
            torch.Tensor: Combined tensor of [token_idx, tf, idf, pos, ner] for each token
            Shape: [seq_len, 5] where each row is [token_idx, tf_score, idf_score, pos_idx, ner_idx]
        """
        # Get base token indices using parent class method
        token_indices = super().encode_sentence(sentence)  # [seq_len]
        
        # Get rich features
        words = preprocess_sentence(sentence)
        doc = self.nlp(sentence)
        
        # Prepare features for each token (excluding BOS/EOS)
        seq_len = len(token_indices)
        features = torch.zeros(seq_len, 5)  # [seq_len, 5] for [token, tf, idf, pos, ner]
        
        # Set token indices in first column
        features[:, 0] = token_indices
        
        # Fill features for actual tokens (excluding BOS/EOS)
        for i, word in enumerate(words, start=1):  # start=1 to skip BOS
            # TF-IDF scores
            features[i, 1] = self.tf.get(word, 0.0)
            features[i, 2] = self.idf.get(word, 0.0)
            
            # POS tag
            if i < len(doc):
                features[i, 3] = self.pos_stoi.get(doc[i-1].pos_, self.unk_idx)
            
            # NER tag (default 0 for non-entities)
            features[i, 4] = 0
        
        # Fill NER tags
        for ent in doc.ents:
            for i in range(ent.start, ent.end):
                features[i+1, 4] = self.ner_stoi.get(ent.label_, self.unk_idx)  # +1 for BOS offset
        #features(S, 5)
        features = features.flatten()
        #features(S*5)
        return features 
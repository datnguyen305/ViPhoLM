import torch
from underthesea import pos_tag, ner
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
        # Remove spacy model loading since we're using underthesea
        
        # Initialize POS and NER mappings
        self.pos_tags = defaultdict(int)
        self.ner_tags = defaultdict(int)
        
        # Compute TF-IDF and create tag mappings
        self.compute_tfidf(config)
        self.create_tag_mappings(config)

    # compute_tfidf method remains the same...

    def create_tag_mappings(self, config):
        """Create mappings for POS and NER tags"""
        json_dirs = [config.path.train]  # Only use training data for tag collection
        
        for json_dir in json_dirs:
            data = json.load(open(json_dir, encoding='utf-8'))
            for item in data.values():
                # Get POS tags using underthesea
                pos_results = pos_tag(item["target"])
                
                # Get NER tags using underthesea
                ner_results = ner(item["target"])
                
                # Collect POS tags
                for word, pos in pos_results:
                    self.pos_tags[pos] += 1
                
                # Collect NER tags
                for word, ner_tag in ner_results:
                    if ner_tag != 'O':  # Only collect named entity tags
                        self.ner_tags[ner_tag] += 1
        
        # Create mappings
        self.pos_stoi = {tag: idx for idx, tag in enumerate(self.pos_tags.keys(), start=4)}
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
        
        # Get POS and NER tags using underthesea
        pos_results = pos_tag(sentence)
        ner_results = ner(sentence)
        
        # Create mapping of word positions to NER tags
        ner_mapping = {}
        for word, tag in ner_results:
            start_pos = sentence.find(word)
            if start_pos != -1:
                ner_mapping[word] = tag
        
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
            if i-1 < len(pos_results):
                pos_tag = pos_results[i-1][1]  # Get POS tag from results
                features[i, 3] = self.pos_stoi.get(pos_tag, self.unk_idx)
            
            # NER tag
            ner_tag = ner_mapping.get(word, 'O')
            features[i, 4] = self.ner_stoi.get(ner_tag, 0)  # Default 0 for non-entities
        
        features = features.flatten()
        return features
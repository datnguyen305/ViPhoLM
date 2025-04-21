import torch
import json
import numpy
import spacy
from collections import Counter, defaultdict
from typing import List
from vocabs.vocab import Vocab
from vocabs.utils import preprocess_sentence
from builders.vocab_builder import META_VOCAB


@META_VOCAB.register()
class VocabWithFeatures(Vocab):
    def __init__(self, config):
        super().__init__(config)  # Gọi hàm khởi tạo của Vocab
        self.build_pos_ner_vocab(config)
        self.compute_tfidf([config.path.train, config.path.dev, config.path.test])

    def compute_tfidf(self, json_dirs):
        self.tf = Counter()  # Tần suất từ (Term Frequency)

        doc_freq = defaultdict(int)
        total_docs = 0

        for json_dir in json_dirs:
            data = json.load(open(json_dir, encoding='utf-8'))
            total_docs += len(data)
            for key in data:
                item = data[key]
                text = f"{item['source']} {item['target']}"
                tokens = set(preprocess_sentence(text))
                self.tf.update(tokens)
                for word in tokens:
                    doc_freq[word] += 1

        self.idf = {}
        self.tfidf = {}
        for word in self.tf:
            self.idf[word] = numpy.log(total_docs / (doc_freq.get(word, 1) + 1e-6))
            self.tfidf[word] = self.tf[word] * self.idf[word]

    def get_tfidf_vector_batch(self, sentences: List[str]):
        batch_tfidf_vectors = []
        for sentence in sentences:
            words = preprocess_sentence(sentence)
            vec = [self.tfidf.get(word, 0.0) for word in words]
            batch_tfidf_vectors.append(vec)
        tfidf_tensor = torch.tensor(batch_tfidf_vectors, dtype=torch.float32)
        return tfidf_tensor

    def build_pos_ner_vocab(self, config):
        nlp = spacy.load("vi_core_news_lg")
        pos_counter = Counter()
        ner_counter = Counter()

        json_dirs = [config.path.train, config.path.dev, config.path.test]
        for json_dir in json_dirs:
            data = json.load(open(json_dir, encoding='utf-8'))
            for key in data:
                item = data[key]
                text = f"{item['source']} {item['target']}"
                doc = nlp(text)

                pos_tags = [token.pos_ for token in doc]
                ner_tags = [ent.label_ for ent in doc.ents]

                pos_counter.update(pos_tags)
                ner_counter.update(ner_tags)

        self.pos_itos = {i: tag for i, tag in enumerate(pos_counter.keys())}
        self.pos_stoi = {tag: i for i, tag in enumerate(pos_counter.keys())}

        self.ner_itos = {i: tag for i, tag in enumerate(ner_counter.keys())}
        self.ner_stoi = {tag: i for i, tag in enumerate(ner_counter.keys())}

    def encode_pos_ner(self, sentences: List[str]):
        nlp = spacy.load("vi_core_news_lg")
        pos_ids_batch = []
        ner_ids_batch = []

        for sentence in sentences:
            doc = nlp(sentence)
            pos_ids = [self.pos_stoi.get(token.pos_, self.unk_idx) for token in doc]
            ner_ids = [0] * len(doc)
            for ent in doc.ents:
                for i in range(ent.start, ent.end):
                    ner_ids[i] = self.ner_stoi.get(ent.label_, self.unk_idx)

            pos_ids_batch.append(pos_ids)
            ner_ids_batch.append(ner_ids)

        pos_tensor = torch.tensor(pos_ids_batch, dtype=torch.long)
        ner_tensor = torch.tensor(ner_ids_batch, dtype=torch.long)
        return pos_tensor, ner_tensor
    
    @property
    def len_pos(self):
        return len(self.pos_stoi)
    
    @property
    def len_ner(self):
        return len(self.ner_stoi)
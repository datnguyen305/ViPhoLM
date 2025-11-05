from builders.vocab_builder import META_VOCAB
from .vocab import Vocab
import torch
from typing import List

@META_VOCAB.register()
class HierarchicalVocab(Vocab):
    def encode_sentences(self, sentences: List[str], max_words: int) -> torch.Tensor:
        """
        Encode a list of sentences into a tensor of shape (S, W).
        - sentences: List of sentences to encode.
        - max_words: Maximum number of words per sentence.
        Returns:
        - Tensor of shape (S, W) where S is the number of sentences and W is max_words.
        """
        encoded_sentences = []
        for sentence in sentences:
            vec = self.encode_sentence(sentence)  # Encode từng câu
            if len(vec) < max_words:
                pad_len = max_words - len(vec)
                vec = torch.cat([vec, torch.full((pad_len,), self.pad_idx, dtype=torch.long)])
            else:
                vec = vec[:max_words]
            encoded_sentences.append(vec)

        return torch.stack(encoded_sentences, dim=0)  # (S, W)
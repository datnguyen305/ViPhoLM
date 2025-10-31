# utils/oov.py
# from typing import List
# from ..data_utils.text_sum_dataset import Vocab

# def get_oovs(batch_sentences: List[List[int]], vocab: Vocab):
#     batch_oovs = []
#     for sent in batch_sentences:
#         oovs = []
#         for token in sent:
#             if token not in vocab.stoi:
#                 if token not in oovs:
#                     oovs.append(token)
#         batch_oovs.append(oovs)
#     max_oov_size = max(len(o) for o in batch_oovs)
#     return batch_oovs, max_oov_size

from typing import List, Tuple
from data_utils.text_sum_dataset import Vocab

def get_oovs(batch_sentences: List[List[int]], vocab) -> Tuple[List[List[int]], int]:
    """
    Extract out-of-vocabulary token indices from batch sentences.
    
    Args:
        batch_sentences: List of sentences, where each sentence is a list of token indices
        vocab: Vocabulary object
        
    Returns:
        batch_oovs: List of OOV token indices for each sentence
        max_oov_size: Maximum number of unique OOVs in any sentence
    """
    batch_oovs = []
    vocab_size = vocab.vocab_size
    
    for sent in batch_sentences:
        oovs = []
        for token in sent:
            # Check if token index is beyond vocabulary size (OOV)
            # chỉ số token > vocab_size
            if token >= vocab_size and token != vocab.pad_idx:
                if token not in oovs:
                    oovs.append(token)
        batch_oovs.append(oovs) # ds 2 chiều
    
    max_oov_size = max(len(o) for o in batch_oovs) if batch_oovs else 0
    return batch_oovs, max_oov_size


def article_oovs_to_extended_vocab(batch_sentences: List[List[int]], batch_oovs: List[List[int]], vocab) -> List[List[int]]:
    """
    Map article tokens to extended vocabulary indices.
    OOV tokens are mapped to vocab_size + oov_position.
    
    Args:
        batch_sentences: Original token indices
        batch_oovs: OOV tokens for each sentence
        vocab: Vocabulary object
        
    Returns:
        Extended vocabulary indices
    """
    extended_batch = []
    vocab_size = vocab.vocab_size
    
    for sent, oovs in zip(batch_sentences, batch_oovs):
        extended_sent = []
        for token in sent:
            if token >= vocab_size and token in oovs:
                # Map to extended vocab: vocab_size + position in oovs list
                oov_idx = oovs.index(token)
                extended_sent.append(vocab_size + oov_idx)
            else:
                extended_sent.append(token)
        extended_batch.append(extended_sent)
    
    return extended_batch
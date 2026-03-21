from torch.utils.data import Dataset
import json 
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
from builders.dataset_builder import META_DATASET
from utils.instance import Instance, InstanceList
from vocabs.vocab_translation import MTVocab
from vocabs.utils import preprocess_sentence 

@META_DATASET.register() 
class MachineTranslationDataset(Dataset):
    def __init__(self, config, vocab: MTVocab):
        super().__init__()
        path: str = config.path
        self.data = json.load(open(path, 'r', encoding='utf-8'))
        self.vocab = vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        key = f"sample_{idx}"
        vi_sentence = item["vietnamese"] # str
        en_sentence = item["english"] # str

        vi_sentence = preprocess_sentence(vi_sentence) # list[str]
        encoded_vi = self.vocab.encode_caption_vietnamese(vi_sentence)

        encoded_en = self.vocab.encode_sentence_english(en_sentence)
        
        return Instance(
            id=key,
            input_vietnamese = encoded_vi, # [(<bos>, <pad>, <pad>), (<initiate>, <rhyme>, <tone>), (<eos>, <pad>, <pad>)]
            input_english = encoded_en # [<bos>, ..., <eos>]
        )
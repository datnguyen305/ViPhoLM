from torch.utils.data import Dataset
import json 
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
from builders.dataset_builder import META_DATASET
from utils.instance import Instance, InstanceList
from vocabs.vocab_translation import MTVocab
from vocabs.utils import preprocess_sentence 


def collate_fn(items: List[Instance]) -> InstanceList:
    pad_idx = 0 
    bos_idx = 1
    eos_idx = 2
    unk_idx = 3

    list_ids = [item.id for item in items] # Lấy list id 
    list_vi = [item.input_vietnamese for item in items]
    list_en = [item.input_english for item in items]
    
    # vietnamese padding
    batch_size = len(list_vi)
    max_len_vi = max(seq.shape[0] for seq in list_vi)
    padded_vi = torch.full((batch_size, max_len_vi, 3), fill_value=pad_idx, dtype=torch.long)
    padded_vi[:, :, 0] = unk_idx
    for i, seq in enumerate(list_vi):
        length = seq.shape[0]
        padded_vi[i, :length, :] = seq    

    # english padding
    padded_en = pad_sequence(list_en, batch_first=True, padding_value=pad_idx)
    
    # 4. Trả về đối tượng InstanceList chứa batch đã pad
    return InstanceList(
        id = list_ids,
        input_vietnamese=padded_vi,
        input_english=padded_en
    )
    

@META_DATASET.register() 
class MachineTranslationDataset(Dataset):
    def __init__(self, config, vocab: MTVocab):
        super().__init__()
        path: str = config.path
        data = json.load(open(path, 'r', encoding='utf-8'))
        self.data = data
        self.vocab = vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sample_id = item.get("id", str(idx))
        vi_sentence = item["vietnamese"] # str
        en_sentence = item["english"] # str

        vi_sentence = preprocess_sentence(vi_sentence) # list[str]
        encoded_vi = self.vocab.encode_caption_vietnamese(vi_sentence)

        encoded_en = self.vocab.encode_sentence_english(en_sentence)
        
        return Instance(
            id=sample_id,
            input_vietnamese = encoded_vi, # [(<bos>, <pad>, <pad>), (<initiate>, <rhyme>, <tone>), (<eos>, <pad>, <pad>)]
            input_english = encoded_en # [<bos>, ..., <eos>]
        )
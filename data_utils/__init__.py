from typing import List
from utils.instance import Instance, InstanceList
from .text_sum_dataset import TextSumDataset
from .text_sum_dataset_bpe import TextSumDatasetBPE
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)
from typing import List
from utils.instance import Instance, InstanceList

from .text_sum_dataset import TextSumDataset
from .text_sum_dataset_hierarchical import HierarchicalTextSumDataset   

def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)

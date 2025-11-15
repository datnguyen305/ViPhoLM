from typing import List
from utils.instance import Instance, InstanceList

from .text_sum_dataset import TextSumDataset
from .text_sum_dataset_hierarchical import TextSumDatasetHierarchical   
from .text_sum_dataset_oov import TextSumDatasetOOV
import torch

def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)

def collate_fn_oov(items: List[Instance]) -> InstanceList:
    """
    Hàm tùy chỉnh để gộp một batch các Instance (đã hỗ trợ PGN).
    """
    
    # 1. Thu thập tất cả dữ liệu từ batch
    ids = []
    input_ids_list = []
    labels_list = []
    shifted_labels_list = []
    oov_lists = [] # List các list string OOV
    
    for instance in items:
        ids.append(instance.id)
        input_ids_list.append(instance.input_ids)
        labels_list.append(instance.label)
        shifted_labels_list.append(instance.shifted_right_label)
        oov_lists.append(instance.oov_list)

    # 2. Padding các tensor (Giả sử 0 là pad_idx)
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=0 
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=0
    )
    shifted_labels_padded = torch.nn.utils.rnn.pad_sequence(
        shifted_labels_list, batch_first=True, padding_value=0
    )

    # 3. Trả về một "Instance" đại diện cho cả batch
    return Instance(
        id = ids,
        input_ids = input_ids_padded,
        label = labels_padded,
        shifted_right_label = shifted_labels_padded,
        oov_list = oov_lists # <-- Giữ nguyên là List[List[str]]
    )
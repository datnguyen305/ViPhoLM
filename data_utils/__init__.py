from typing import List
from utils.instance import Instance, InstanceList

from .text_sum_dataset import TextSumDataset
from .text_sum_dataset_phoneme import TextSumDatasetPhoneme
# from .text_sum_dataset_hierarchical import TextSumDatasetHierarchical   
from .text_sum_dataset_oov import TextSumDatasetOOV
from .text_sum_dataset_seneca import TextSumDatasetSeneca
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)

def collate_fn_oov(items: List[Instance]) -> Instance:
    """
    Collate function MỚI, được thiết kế để hoạt động với
    TextSumDatasetOOV (loại Dataset đã xử lý PGN).
    """
    PAD_IDX = 0
    
    # 1. Thu thập tất cả các tensor và list từ batch
    ids = []
    input_ids_list = []
    labels_list = []
    shifted_labels_list = []
    oov_lists = []
    max_oov_count = 0

    for instance in items:
        ids.append(instance.id)
        # 'instance.input_ids' BÂY GIỜ LÀ TENSOR
        input_ids_list.append(instance.input_ids)
        labels_list.append(instance.label)
        shifted_labels_list.append(instance.shifted_right_label)
        
        # 'instance.oov_list' LÀ List[str]
        oov_lists.append(instance.oov_list)
        if len(instance.oov_list) > max_oov_count:
            max_oov_count = len(instance.oov_list)

    # 2. Pad các chuỗi tensor
    input_ids_padded = pad_sequence(
        input_ids_list, batch_first=True, padding_value=PAD_IDX
    )
    labels_padded = pad_sequence(
        labels_list, batch_first=True, padding_value=PAD_IDX
    )
    shifted_labels_padded = pad_sequence(
        shifted_labels_list, batch_first=True, padding_value=PAD_IDX
    )

    # 3. Tạo 'extra_zeros' (thứ duy nhất còn thiếu)
    batch_size = len(items)
    extra_zeros = torch.zeros((batch_size, max_oov_count))

    # 4. Trả về Instance đã gộp
    # Chú ý: input_ids_padded chính là extended_source_idx
    return Instance(
        id = ids,
        input_ids = input_ids_padded,
        label = labels_padded,
        shifted_right_label = shifted_labels_padded,
        oov_list = oov_lists,
        
        # Thêm các trường mà Task/Model cần
        extended_source_idx = input_ids_padded,
        extra_zeros = extra_zeros
    )
def collate_fn_seneca(batch):
    """
    batch: List[Instance]
    """

    pad_idx = batch[0].vocab.pad_idx if hasattr(batch[0], "vocab") else 0

    # -------- sentences --------
    batch_sentences = []
    max_sent = max(len(x.sentences) for x in batch)
    max_len = max(
        max(len(s) for s in x.sentences) for x in batch
    )

    for item in batch:
        sent_tensors = []
        for s in item.sentences:
            if isinstance(s, torch.Tensor):
                s = s.clone().detach()
            else:
                s = torch.tensor(s, dtype=torch.long)
            if len(s) < max_len:
                s = F.pad(s, (0, max_len - len(s)), value=pad_idx)
            sent_tensors.append(s)

        # pad number of sentences
        if len(sent_tensors) < max_sent:
            pad_sent = torch.full(
                (max_len,), pad_idx, dtype=torch.long
            )
            sent_tensors.extend(
                [pad_sent] * (max_sent - len(sent_tensors))
            )

        batch_sentences.append(torch.stack(sent_tensors))

    sentences = torch.stack(batch_sentences)  # (B, N, L)

    # -------- entities (keep as list) --------
    entities = [x.entities for x in batch]

    # -------- target --------
    labels = [
        x.label.clone().detach()
        if isinstance(x.label, torch.Tensor)
        else torch.tensor(x.label, dtype=torch.long)
        for x in batch
    ]

    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=pad_idx
    )

    return {
        "sentences": sentences,
        "entities": entities,
        "label": labels,
        "id": [x.id for x in batch]
    }


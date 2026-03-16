import torch

def create_padding_mask(seq, pad_idx):
    """
    Tạo attention_mask chuẩn cho Longformer.
    Shape: (Batch_Size, Seq_Len)
    Giá trị:
      -1: No attention (Dành cho Padding)
       0: Local attention (Cửa sổ trượt Sliding window)
       1: Global attention (Dành cho token đặc biệt gom ngữ cảnh)
    """
    
    mask = torch.zeros_like(seq, dtype=torch.long)
    
    is_pad = (seq == pad_idx)
    mask[is_pad] = -1
    
    mask[:, 0] = 1
    
    return mask

def create_standard_padding_mask(seq, pad_idx):
    """ Dùng cho PyTorch MultiheadAttention truyền thống (Trả về True/False) """
        
    # Trả về Boolean: True là Pad (Che đi), False là chữ thật (Giữ lại)
    return (seq == pad_idx)
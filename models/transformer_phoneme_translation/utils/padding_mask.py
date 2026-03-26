def create_padding_mask(seq, pad_idx=0):
    """
    Hỗ trợ cả Normal (B, S) và Phoneme (B, S, 3).
    Trả về Mask dạng (B, S): True tại vị trí là Pad.
    """
    # Nếu seq có 3 chiều (Phoneme), lấy slice đầu tiên: (B, S, 3) -> (B, S)
    _seq = seq[:, :, 0] if seq.dim() == 3 else seq
    return (_seq == pad_idx)
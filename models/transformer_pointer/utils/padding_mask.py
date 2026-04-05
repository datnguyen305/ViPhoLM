def create_padding_mask(seq, pad_idx):
    """
    Tạo mask cho key_padding_mask (True là Pad).
    Shape: (Batch_Size, Seq_Len)
    """
    return (seq == pad_idx)
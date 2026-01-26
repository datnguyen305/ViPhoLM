def create_padding_mask(seq, pad_idx):
    # seq: (B, S)
    # mask: (B, S), True tại vị trí từ thật, False tại vị trí <pad>
    mask = (seq != pad_idx) 
    
    # Expand chiều để broadcast với scores (B, H, S, S)
    # Kết quả: (B, 1, 1, S)
    return mask.unsqueeze(1).unsqueeze(2)
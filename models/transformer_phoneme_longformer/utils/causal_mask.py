import torch
def create_causal_mask(seq_len, device):
        """
        Tạo mask Causal dạng Boolean để khớp với padding_mask.
        Logic: True = Che (Ignore), False = Nhìn (Keep).
        """
        # Tạo ma trận True ở tam giác trên (vị trí tương lai cần che)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        return mask
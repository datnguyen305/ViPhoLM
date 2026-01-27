
import torch
def create_causal_mask(seq, device):
    # 1. Tạo ma trận vuông (size x size) toàn số 1
    # 2. torch.tril giữ lại tam giác dưới (triangular lower), xóa tam giác trên thành 0
    mask = torch.tril(torch.ones((seq, seq), device=device))
    
    # 3. Đưa về dạng (1, 1, S, S) để tương thích với scores (B, H, S, S)
    return mask.unsqueeze(0).unsqueeze(0).bool()
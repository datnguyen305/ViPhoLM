import torch
import torch.nn as nn

class Phrasal_Lexeme(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size, bias = False)

    def forward(self, query, key):
        # query: (B, S, D)
        # key: (B, S, D)
        Sk = key.size(1)
        Sq = query.size(1)

        if Sq != Sk:
            raise ValueError(
                f"Phrasal_Lexeme module chỉ hỗ trợ ma trận vuông (Self-Attention). "
                f"Nhưng hiện tại nhận được Sq={Sq} và Sk={Sk}. "
                f"Hãy kiểm tra xem bạn có đang vô tình gọi module này trong Cross-Attention của Decoder không."
            )
        intermediate = self.linear(query)  # (B, S, D)

        # intermediate: (B, S, D)
        # key.transpose: (B, D, S)
        scores = torch.matmul(intermediate, key.transpose(-2, -1))  # (B, S, S)
        r_left = torch.diagonal(scores, offset=-1, dim1=-2, dim2=-1) # (B, S-1)
        r_right = torch.diagonal(scores, offset=1, dim1=-2, dim2=-1) # (B, S-1)
        left_side = r_left[:, :-1]
        right_side = r_right[:, 1:] 

        comparison = torch.stack([left_side, right_side], dim=-1) # (B, S-2, 2)
        pr = torch.softmax(comparison, dim=-1) # (B, S-2, 2)
        """
            => pr[:,i,0] left_side
            => pr[:,i,1] right_side
            dim = (B, S-2)

        """
        # Pi = sqrt(left_side * right_side)
        pr_i_right = pr[:, :-1, 1] 
        pr_iplus1_left = pr[:, 1:, 0]
        Pi = torch.sqrt(pr_i_right * pr_iplus1_left + 1e-9)  
        # Pi: (B, S-2)
        log_P = torch.log(Pi + 1e-8)  # tránh log(0)
        # log_P: (B, S-2)
        Pi_j = torch.exp(torch.sum(log_P, dim=-1)) 
        
        # Pi_j: (B)
        return Pi_j
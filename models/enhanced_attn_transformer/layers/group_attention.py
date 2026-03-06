import torch 
from torch import nn
from torch.nn import functional as F

class GroupAttention(nn.Module):
    def __init__(self, config):
        super(GroupAttention, self).__init__()
        self.h = config.head
        self.d_k = config.d_model // config.head
        self.linear_key = nn.Linear(self.d_k, self.d_k)
        self.linear_query = nn.Linear(self.d_k, self.d_k)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, context, eos_mask, prior):
        bs, seq_len = context.size()[:2]

        context = self.norm(context).view(bs, seq_len, self.h, self.d_k).transpose(1, 2)

        a = torch.diag(torch.ones(seq_len - 1), 1).long().to(context.device)
        b = torch.diag(torch.ones(seq_len), 0).long().to(context.device)
        c = torch.diag(torch.ones(seq_len - 1), -1).long().to(context.device)

        mask = torch.logical_and(eos_mask, (a+c))
        
        key = self.linear_key(context)
        query = self.linear_query(context)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k
        
        scores = scores.masked_fill(mask == 0, -1e4)
        neibor_attn = F.softmax(scores, dim = -1)
        neibor_attn = torch.sqrt(neibor_attn*neibor_attn.transpose(-2,-1) + 1e-4)
        neibor_attn = prior + (1. - prior)*neibor_attn

        tri_matrix = torch.triu(torch.ones(seq_len, seq_len), diagonal = 0).float().to(context.device)
        tri_matrix = tri_matrix.unsqueeze(0).unsqueeze(0)
        t = torch.log(neibor_attn + 1e-9).masked_fill(a == 0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b == 0, 1e-4)
        
        return g_attn, neibor_attn
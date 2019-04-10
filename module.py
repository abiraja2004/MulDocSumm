import math

import torch
import torch.nn as nn


def dot_attention(value_dict, dropout=None):
    """
    values_dict: dict of (field, value) pairs
    value: (B, latent_dim) or (B, hidden_dim * 2)
    """
    value = torch.stack([v for v in value_dict.values()], dim=1) # (B, 5, *)
    dim = value.size(-1)
    # scores:(B, 5, 5)
    scores = torch.matmul(value, value.transpose(-1, -2)) / math.sqrt(dim)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)#, p_attn # (B, 5, *)


class proj_attention(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = nn.Linear(dimension, 1)

    def forward(self, value_dict):
       value = torch.stack([v for v in value_dict.values()], dim=1) # (B, 5, *)
       p_attn = self.linear(value).softmax(dim=1) # (B, 5, 1)
       return torch.matmul(p_attn.transpose(-1, -2), value) # (B, 1, *)


def get_attention(type, dimension):
    if type =='proj':
        return proj_attention(dimension)
    elif type =='dot':
        return dot_attention


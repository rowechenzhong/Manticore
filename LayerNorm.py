import torch
import numpy as np


class LayerNorm(torch.nn.Module):
    def __init__(self, size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a = torch.nn.Parameter(torch.ones(size))
        self.b = torch.nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        """
        This function takes in the input and returns the output after
        applying the layer normalization to the last dimension.
        input: (batch_size, seq_len, size)
        output: (batch_size, seq_len, size)
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

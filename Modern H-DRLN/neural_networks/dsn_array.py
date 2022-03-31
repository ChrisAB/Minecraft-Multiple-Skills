import torch
import torch.nn as nn
import torch.nn.functional as F


class DSNArray(nn.Module):
    def __init__(self, array_of_DSN):
        super(DSNArray, self).__init__()

        self.linears = nn.ModuleList(array_of_DSN)

    def forward(self, x):
        return [linear(x) for linear in self.linears]

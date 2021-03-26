
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import truncnorm
from utils import truncated_normal
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, hiddenWidth1=100, hiddenWidth2=64, outputWidth=5, weightInit=-1):
        super(QNetwork, self).__init__()

        # Hidden Layer 1
        self.W_hidden1 = nn.Parameter(
            truncated_normal.truncated_normal(torch.ones((512, hiddenWidth1)), std=0.1), requires_grad=True)
        self.b_hidden1 = nn.Parameter(
            torch.full(size=torch.Size([hiddenWidth1]), fill_value=0.1), requires_grad=True)
        self.act_hidden1 = torch.nn.ReLU()

        # Hidden Layer 2
        self.W_hidden2 = nn.Parameter(torch.FloatTensor(
            100, hiddenWidth2).uniform_(weightInit, 1), requires_grad=True)
        self.b_hidden2 = nn.Parameter(torch.FloatTensor(
            hiddenWidth2).uniform_(weightInit, 1), requires_grad=True)
        self.act_hidden2 = torch.nn.ReLU()

        self.W_output = nn.Parameter(truncated_normal.truncated_normal(
            torch.ones((100, outputWidth)), std=0.1), requires_grad=True)
        self.b_output = nn.Parameter(torch.full(
            size=torch.Size([outputWidth]), fill_value=0.1), requires_grad=True)

    def forward(self, x):
        y_hidden1 = torch.add(torch.matmul(
            x, self.W_hidden1), self.b_hidden1)
        act_hidden1 = self.act_hidden1(y_hidden1)

        y_hidden2 = torch.add(torch.matmul(
            act_hidden1, self.W_hidden2), self.b_hidden2)
        act_hidden2 = self.act_hidden2(y_hidden2)

        return torch.add(torch.matmul(act_hidden1, self.W_output), self.b_output)

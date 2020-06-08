# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.lin_1 = nn.Linear(in_features=784, out_features=128)
        self.lin_2 = nn.Linear(in_features=128, out_features=64)
        self.lin_3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        x = F.log_softmax(self.lin_3(x), dim=1)
        return x
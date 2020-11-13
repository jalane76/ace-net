# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class neural_network(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(neural_network, self).__init__()
        self.hidden = nn.Linear(num_input, num_hidden)
        self.out = nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.out(x)


class network_wrapper(nn.Module):
    def __init__(self, model_to_wrap):
        super(network_wrapper, self).__init__()
        self.wrapped_model = model_to_wrap

    def forward(self, x):
        return F.softmax(self.wrapped_model(x), dim=-1)


def wrap_model_activation(model):
    return network_wrapper(model)

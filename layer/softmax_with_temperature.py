import torch
import torch.nn as nn


class SoftmaxWithTemperature(nn.Module):
    def __init__(self, T):
        super(SoftmaxWithTemperature, self).__init__()
        assert (type(T) == float or type(T) == int) and T > 0, "T must be a positive number."
        self.T = T

    def forward(self, x):
        return (x / self.T).exp() / (x / self.T).exp().sum(-1).unsqueeze(-1)

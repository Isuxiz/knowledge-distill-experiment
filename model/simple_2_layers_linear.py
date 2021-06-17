import torch
import torch.nn as nn
from collections import OrderedDict
from layer.softmax_with_temperature import SoftmaxWithTemperature


class SimpleLinear(nn.Module):
    def __init__(self, T):
        super(SimpleLinear, self).__init__()
        self.linear_net = torch.nn.Sequential(
            OrderedDict([
                ('flatten', nn.Flatten()),
                ('l1', nn.Linear(32 * 32 * 1, 16)),
                ('relu1', nn.ReLU()),
                ('l2', nn.Linear(16, 10)),
                ('relu2', nn.ReLU()),
            ])
        )
        self.softmax_with_T = SoftmaxWithTemperature(T)
        self.softmax_without_T = nn.Softmax()

    def forward(self, x):
        x = self.linear_net(x)
        x_soft = self.softmax_with_T(x)
        x_hard = self.softmax_without_T(x)
        return torch.stack((x_soft, x_hard))


def get_simple_2_layers_linear(T):
    return SimpleLinear(T)

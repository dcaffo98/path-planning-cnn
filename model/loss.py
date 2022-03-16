# import torch
from torch import nn
from torch.nn import MSELoss
import torch


class MSEMapLoss(nn.Module):
    def __init__(self):
        super(MSEMapLoss, self).__init__()
        self._loss = MSELoss()

    def forward(self, x, y):
        return self._loss(x, y)
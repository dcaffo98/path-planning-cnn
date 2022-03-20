import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, max_h, max_w):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_w).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, max_h, 2) * (-math.log(10000.0) / max_w))
        pe = torch.zeros(1, 1, max_h, max_w)
        pe[:, :, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2)]

        
class GaussianRelativePE(nn.Module):
    def __init__(self, side):
        super().__init__()
        self.sigma_square = (side / 5) ** 2
        self.alpha = 1 / (2 * math.pi * self.sigma_square)
        self.side = side
        coord_r = torch.stack([torch.arange(side) for _ in range(side)])
        coord_c = coord_r.T
        self.register_buffer('coord_r', coord_r)
        self.register_buffer('coord_c', coord_c)

    def forward(self, x, center):
        pe = self.alpha * torch.exp(- ((self.coord_r.view(1, self.side, self.side) - center[:, 0:1].unsqueeze(1)) ** 2 + \
            (self.coord_c.view(1, self.side, self.side) - center[:, 1:2].unsqueeze(1)) ** 2) / (2 * self.sigma_square))
        pe /= pe.sum(dim=(-1, -2)).view(-1, 1, 1)
        return x + pe.unsqueeze(1)


if __name__ == '__main__':
    side = 100
    bsz = 4
    grpe = GaussianRelativePE(side)
    map = torch.rand((bsz, 1, side, side))
    goal = torch.randint(0, side, (bsz, 2), dtype=torch.long)
    out = grpe(map, goal)
    pass
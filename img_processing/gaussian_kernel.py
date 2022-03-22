import numpy as np

def get_gaussian(k=3, sigma=0, normalized=True):
    if sigma == 0:
        sigma = k / 5
    sigma_square = sigma ** 2
    coord_x = np.stack([np.arange(-(k // 2), k // 2 + 1) for _ in range(k)])
    coord_y = coord_x.T
    alpha = 1 / (2 * np.pi * sigma_square)
    out = alpha * np.exp(- 1 / (2 * sigma_square) * (coord_x ** 2  + coord_y ** 2))
    if normalized:
        out /= out.sum()
    return out
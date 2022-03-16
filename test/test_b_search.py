from dataset.map_sample import MapSample
from dataset.utils import b_search
import torch

if __name__ == '__main__':
    sample = MapSample.load('map_dataset/test/795f8455-0202-4c99-ae37-8588e6421176.pt')
    map = torch.zeros_like(sample.map)
    map[sample.path[:, 0], sample.path[:, 1]] = 1
    out = b_search(map, sample.start, sample.goal, True)
    assert not torch.any(out != sample.path)
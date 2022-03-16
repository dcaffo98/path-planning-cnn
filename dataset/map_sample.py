import numpy as np
import torch
import uuid


WHITE = np.array((255, 255, 255), dtype=np.uint8)
RED = np.array((0, 0, 255), dtype=np.uint8)
GREEN = np.array((0, 255, 0), dtype=np.uint8)
BLUE = np.array((255, 0, 0), dtype=np.uint8)


class MapSample(object):
    def __init__(self, map, start, goal, path, device=None):
        super(MapSample, self).__init__()
        if device is None:
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device
        self.map = torch.tensor(map, dtype=torch.float32, device=self._device)
        self.start = torch.tensor(start, dtype=torch.long, device=self._device)
        self.goal = torch.tensor(goal, dtype=torch.long, device=self._device)
        self.path = torch.tensor(path, dtype=torch.long, device=self._device)

    def to(self, device):
        self._device = device
        self.map = self.map.to(device)
        self.start = self.start.to(device)
        self.goal = self.goal.to(device)
        self.path = self.path.to(device)

    def save(self, path=None):
        self.to('cpu')
        if path is None:
            path = str(uuid.uuid4()) + '.pt'
        torch.save(self, path)

    @staticmethod
    def load(path):
        try:
            sample = torch.load(path)
        except IOError as e:
            print(e)
            sample = None
        return sample

    def bgr_map(self, start_color=RED, goal_color=BLUE, path_color=GREEN):
        map_np, start_np, goal_np, path_np = self.numpy()
        return MapSample.get_bgr_map(map_np, start_np, goal_np, path_np, start_color, goal_color, path_color)

    def numpy(self):
        return self.map.cpu().detach().numpy(), self.start.cpu().detach().numpy(), self.goal.cpu().detach().numpy(), self.path.cpu().detach().numpy()

    @staticmethod
    def get_bgr_map(map, start, goal, path, start_color=RED, goal_color=BLUE, path_color=GREEN, remove_first_path=True):
        h, w = map.shape
        if remove_first_path:
            path = path[1:]
        if type(path) == list or type(path) == tuple:
            path = np.array(path)
        bgr_map = np.zeros((h, w, 3), dtype=np.uint8)
        idx = np.argwhere(map > 0).reshape(-1, 2)
        bgr_map[idx[:, 0], idx[:, 1]] = WHITE
        if np.any(path):
            bgr_map[path[:, 0], path[:, 1]] = path_color
        bgr_map[start[0], start[1]] = start_color
        bgr_map[goal[0], goal[1]] = goal_color
        return bgr_map


if __name__ == '__main__':
    import cv2
    sample = MapSample.load('map_dataset/479e8ea0-b439-4607-8f15-23e2dad42cab.pt')
    color_map = sample.bgr_map()
    cv2.imshow('map', cv2.resize(color_map, (600, 600)))
    cv2.waitKey(0)
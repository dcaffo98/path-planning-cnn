from dataset.map_sample import GREEN, RED, BLUE, WHITE
import numpy as np
import cv2

PATH = 'imgs/maps/map_with_pos.png'


def fill_map(map, bin_size_h=3, bin_size_w=3, iterations=1, set_inf=True):
    kernel = np.ones((bin_size_h, bin_size_w), dtype=np.uint8)
    filled_map = np.where(map > 0, 1, 0)
    filled_map = cv2.dilate(map.astype(np.float32), kernel, iterations=iterations)
    if set_inf:
        filled_map[filled_map > 0] = np.inf
    return filled_map

def parse_bgr(bgr, obst_color=WHITE, path_color=GREEN, start_color=RED, goal_color=BLUE):
    path = np.argwhere(np.all(bgr == path_color, axis=2)).squeeze()
    start = np.argwhere(np.all(bgr == start_color, axis=2)).squeeze()
    goal = np.argwhere(np.all(bgr == goal_color, axis=2)).squeeze()
    map = np.where(np.all(bgr == obst_color, axis=2), 1.0, 0)
    if np.any(start):
        map[start[0], start[1]] = 0
    if np.any(goal):
        map[goal[0], goal[1]] = 0
    if np.any(path):
        if len(path.shape) > 1:
            start = path[0]
        else:
            start = np.array((path[0], path[1]))    
        map[path[..., 0], path[..., 1]] = 0
    return map, start, goal, path
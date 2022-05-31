import numpy as np
import cv2
from random import choice
try:
    from planning.cpp_dstar_lite import DStarLite
except ImportError as e:
    print(e)
    print('Cannot import D* Lite implementation in c++. Let\'s try with the python one.')
    from planning.dstar_lite import DStarLite
from dataset.map_sample import MapSample


class MapFactory(object):
    MAZE_COMBINATIONS = (
        (7, 0.06),
        (7, 0.07),
        (7, 0.08),
        (7, 0.09),
        (5, 0.10),
        (5, 0.11),
        (5, 0.12),
        (5, 0.13),
        (5, 0.14),
        # (3, 0.35),
        # (3, 0.36),
        # (3, 0.37),
        # (3, 0.38),
        # (3, 0.39),
        # (3, 0.40),
    )

    @staticmethod
    def random_maze(h=100, w=100, k=7, alpha=0.06):
        maze = np.random.rand(h, w)
        maze = np.where(maze > alpha, 1, 0)
        kernel = np.ones((k, k), np.uint8)
        maze = cv2.morphologyEx(maze.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
        return maze

    @classmethod
    def random_maze_(cls, h=100, w=100):
        k, alpha = choice(cls.MAZE_COMBINATIONS)
        return MapFactory.random_maze(h, w, k, alpha)
    
    @staticmethod
    def random_start_goal(maze, min_dist):
        h, w = maze.shape
        max_it = h * w
        start, goal = np.array([[0], [0]], dtype=np.int64), np.array([[0], [0]], dtype=np.int)
        i = 0
        while np.linalg.norm(start - goal) < min_dist and i < max_it:
            x = np.random.randint(0, h, size=(2, 1), dtype=np.int64)   # rows
            y = np.random.randint(0, w, size=(2, 1), dtype=np.int64)   # cols
            start = np.concatenate((x[0], y[0]))
            goal = np.concatenate((x[1], y[1]))
            if maze[start[0], start[1]] > 0:
                start = goal
            i += 1
        if i == max_it:
            raise ValueError('Cannot find start and goal with enough distance')
        return start, goal

    @staticmethod
    def solve(maze, start, goal, max_it, obst_margin, goal_margin):
        maze = maze.astype(np.float64)
        maze[maze > 0] = np.inf
        solved, path = False, []
        try:
            ds = DStarLite(maze, goal[0], goal[1], start[0], start[1], max_it, False, obst_margin, goal_margin)
            next_step = None
            while next_step is not None or not path:
                next_step = ds.step()
                path.append(next_step)
            if path and path[-1] is None:
                path.pop()
            if len(path) > 1:
                solved = True
        except ValueError:
            if len(path) <= 1:
                print("Unfeasible instance")
            else:
                # solved at the best
                solved = True
        return solved, path

    @staticmethod
    def make_sample_(h, w, min_dist):
        map = MapFactory.random_maze_(h, w)
        start, goal = MapFactory.random_start_goal(map, min_dist)
        return map, start, goal

    @staticmethod
    def make_sample(h, w, min_dist, max_it, obst_margin, goal_margin):
        sample = None
        map, start, goal = MapFactory.make_sample_(h, w, min_dist)
        solved, path = MapFactory.solve(map, start, goal, max_it, obst_margin, goal_margin)
        if solved:
            sample = MapSample(map, start, goal, path)
        return solved, sample


if __name__ == '__main__':
    solved, sample = False, None
    while not solved:
        solved, sample = MapFactory.make_sample(100, 100, 70, 10000, 0, 1)
    bgr = sample.bgr_map()
    cv2.imshow('map', cv2.resize(bgr, (500, 500)))
    cv2.waitKey(0)

import os
import uuid
from math import sqrt
from random import uniform
from multiprocessing import Process
from dataset.map_factory import MapFactory


PATH = 'map_dataset'

def generate_sample(h, w, min_dist, max_ds_it, obst_margin, goal_margin, location=PATH):
    feasible = False
    while not feasible:
        feasible, sample = MapFactory.make_sample(h, w, min_dist, max_ds_it, obst_margin, goal_margin)
    filename = os.path.join(location, str(uuid.uuid4()) + '.pt')
    sample.save(filename)

def generate_random_samples(h, w, min_dist_th, max_ds_it, obst_margin, goal_margin, n=10000, location=PATH):
    max_dist = sqrt(h ** 2 + w ** 2) - 1e-3
    for i in range(n):
        try:
            min_dist = uniform(min_dist_th, max_dist)
            generate_sample(h, w, min_dist, max_ds_it, obst_margin, goal_margin, location)
            print('##### ', i, ' #####')
        except ValueError as e:
            print(e)
            continue


class RandomMapGenerator(object):
    def __init__(self, h, w, min_dist_th, max_ds_it, obst_margin, goal_margin):
        super(RandomMapGenerator, self).__init__()
        self.h = h
        self.w = w
        self.min_dist_th = min_dist_th
        self.max_ds_it = max_ds_it
        self.obst_margin = obst_margin
        self.goal_margin = goal_margin

    def execute(self, n=10000, n_process=4, location=PATH):
        if not os.path.exists(os.path.abspath(location)):
            os.mkdir(os.path.abspath(location))
        for _ in range(n_process):
            Process(target=generate_random_samples, args=(self.h, self.w, self.min_dist_th, self.max_ds_it, 
                self.obst_margin, self.goal_margin, n, location)).start()


if __name__ == '__main__':
    rmg = RandomMapGenerator(100, 100, 20, 10000, 1, 0)
    rmg.execute(20000, n_process=2)

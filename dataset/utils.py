import re
import os
import shutil
import torch
import numpy as np
from shutil import move


PATTERN = '.*(.+\.pt)$'

def move_data(datapath, train_p=None, val_p=None, test_p=None, train_name='train', test_name='test', validation_name='validation'):
    datapath = os.path.abspath(datapath)
    assert abs(1 - train_p - val_p - test_p) < 1e-5, 'Percentage must sum to 1'
    files = [f for f in os.listdir(datapath) if re.search(PATTERN, f) is not None and os.path.isfile(os.path.join(datapath, f))]
    train_id = int(round(len(files) * train_p))
    val_id = int(round(len(files) * (train_p + val_p)))
    if train_id < 1 or val_id < 1:
        # too few data, move everything to train split
        train_id = val_id = len(files)
    train = files[:train_id]
    val = files[train_id:val_id]
    test = files[val_id:]
    assert len(train) + len(val) + len(test) == len(files), 'Error in splitting data'
    dirnames = (train_name, validation_name, test_name)
    splits = (train, val, test)
    for i, dir in enumerate(dirnames):
        dirpath = os.path.join(datapath, dir)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        for f in splits[i]:
            shutil.move(os.path.join(datapath, f), os.path.join(dirpath, f))

def chunk_data(datapath, chunk_size=1000):
    datapath = os.path.abspath(datapath)
    basename = 'data_'
    chunks = []
    chunk_count = 0
    for entry in os.scandir(datapath):
        if os.path.isfile(entry.path):
            chunks.append(entry.name)
        if len(chunks) == chunk_size:
            dirname = basename + str(chunk_count)
            os.mkdir(os.path.join(datapath, dirname))
            for chunk in chunks:
                move(os.path.join(datapath, chunk), os.path.join(datapath, dirname, chunk))
            chunk_count += 1
            chunks = []
    dirname = basename + str(chunk_count)
    os.mkdir(os.path.join(datapath, dirname))
    for chunk in chunks:
                move(os.path.join(datapath, chunk), os.path.join(datapath, dirname, chunk))            

def collect_files(datapath):
    for entry in os.scandir(os.path.abspath(datapath)):
        if os.path.isfile(entry.path):
            yield entry.path
        elif os.path.isdir(entry.path):
            for nested_entry in collect_files(entry.path):
                yield nested_entry

def custom_collate_fn(batch):
    maps = torch.stack([sample.map for sample in batch]).unsqueeze(1)
    starts = torch.stack([sample.start for sample in batch])
    goals = torch.stack([sample.goal for sample in batch])
    paths = []
    for sample in batch:
        path = torch.zeros_like(sample.map, dtype=sample.map.dtype)
        path[sample.path[:, 0], sample.path[:, 1]] = 1.0
        paths.append(path)
    paths = torch.stack(paths)
    return maps, starts, goals, paths

def get_grid(h, w, to_torch=True):
    coords = np.mgrid[0:h, 0:w].reshape(2, -1)
    grid = np.stack((coords[0], coords[1]), axis=1)
    if to_torch:
        grid = torch.tensor(grid)
    return grid

def get_coords_from_idx(r, c, h, w, idx):
    grid = np.mgrid[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)]
    return np.stack((grid[0].ravel(), grid[1].ravel())).T[idx]

def b_search(map, start, goal, to_torch=True):
    h, w = map.shape
    f_map = map
    b_map = f_map.clone()
    start, goal = start.tolist(), goal.tolist()
    f_set, b_set = set(), set()
    f_set.add(str(start))
    b_set.add(str(goal))
    f_path, b_path = [start], [goal]
    f_map[start[0], start[1]] = 0
    b_map[goal[0], goal[1]] = 0
    path = []
    f_stop, b_stop = False, False
    for i in range(h * w):
        if i % 2 == 0 and not f_stop:
            # forward search
            idx = torch.argmax(f_map[max(0, start[0] - 1):start[0] + 2, max(0, start[1] - 1):start[1] + 2]).item()
            f_next = get_coords_from_idx(start[0], start[1], h, w, idx).tolist()
            if str(f_next) in b_set:
                idx = [i for i, pt in enumerate(b_path) if pt == f_next][0]
                path = f_path + b_path[idx::-1]
                break
            elif str(f_next) in f_set and not (f_map[max(0, start[0] - 1):start[0] + 2, max(0, start[1] - 1):start[1] + 2] != 0).any():
                f_stop = True
                continue
            f_path.append(f_next)
            f_set.add(str(f_next))
            start = f_next
            f_map[start[0], start[1]] = 0
        elif not b_stop:
            # backward search
            idx = torch.argmax(b_map[max(0, goal[0] - 1):goal[0] + 2, max(0, goal[1] - 1):goal[1] + 2]).item()
            b_next = get_coords_from_idx(goal[0], goal[1], h, w, idx).tolist()
            if str(b_next) in f_set:
                idx = [i for i, pt in enumerate(f_path) if pt == b_next][0]
                path = f_path[:idx + 1] + b_path[::-1]
                break
            elif str(b_next) in b_set and not (b_map[max(0, goal[0] - 1):goal[0] + 2, max(0, goal[1] - 1):goal[1] + 2] != 0).any():
                b_stop = True
                continue
            b_path.append(b_next)
            b_set.add(str(b_next))
            goal = b_next
            b_map[goal[0], goal[1]] = 0
        elif not f_stop and not b_stop:
            break
    if to_torch:
        path = torch.tensor(path)
    return path


if __name__ == '__main__':
    move_data('map_dataset', 0.35, 0.05, 0.6)

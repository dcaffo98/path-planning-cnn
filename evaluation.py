import torch
import numpy as np
import os
from dataset.map_dataset import MapDataset
from dataset.map_sample import MapSample
from model.ppcnet import PPCNet
from dataset.utils import b_search, custom_collate_fn_extended
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from model.checkpoint import Checkpoint
import pandas as pd


TEST = 'map_dataset_test_of/test'
CHECKPOINT = 'checkpoint_new_min_23.pt'
RESULTS_PATH = 'model_results_evaluation'

if __name__ ==  '__main__':
    if not os.path.exists(os.path.abspath(RESULTS_PATH)):
        os.mkdir(os.path.abspath(RESULTS_PATH))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = PPCNet().to(device)
    if CHECKPOINT:
        Checkpoint.load_checkpoint(CHECKPOINT, model)
    dataset = MapDataset(TEST, lazy=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, collate_fn=custom_collate_fn_extended, pin_memory=True)
    model.eval()
    solved_samples_count = 0
    valid_solved_samples_count = 0
    sep = np.zeros((100, 50, 3), dtype=np.uint8)
    sep[:, 20:31] = np.array((0, 0, 0))
    errors = []
    r_index = []
    with torch.no_grad():
        for i, (map, start, goal, path, filenames, path_array) in enumerate(dataloader):
            if start.dtype != torch.long:
                start, goal = start.long(), goal.long()
            if device != 'cpu':
                map, start, goal, path = map.to(device), start.to(device), goal.to(device), path.to(device)
            out = model(map, start, goal).squeeze()
            r_index.extend([abs_name.split(os.sep)[-1] for abs_name in filenames])
            for j, (filename, model_out) in enumerate(zip(filenames, out)):
                # TODO: add explicit path to custom collate function and use last gt path as goal for bsearch function
                model_path = b_search(model_out, start[j], path_array[j][-1], to_torch=True)
                sample_summary = [int(path[j].sum())]
                if model_path.any():
                    solved_samples_count += 1
                    fname = filename.split(os.sep)[-1].split('.')[0]
                    np_map = map[j].squeeze().cpu().detach().numpy()
                    obstacle_hit = int((np_map[model_path[:, 0], model_path[:, 1]] > 0).sum())
                    if obstacle_hit == 0:
                        valid_solved_samples_count += 1
                    else:
                        fname = '_' + fname
                    sample_summary.extend([len(model_path), obstacle_hit])
                    model_solution = MapSample.get_bgr_map(np_map, start[j], goal[j], model_path.numpy())
                    gt_solution = MapSample.get_bgr_map(np_map, start[j], goal[j], torch.nonzero(path[j] > 0).cpu().numpy())
                    # LEFT img: model solution ------ RIGHT img: dstar solution
                    comparison = np.concatenate((model_solution, sep, gt_solution), axis=1)
                    save_path = os.path.join(os.path.abspath(RESULTS_PATH), fname + '.png')
                    cv2.imwrite(save_path, cv2.cvtColor(cv2.resize(comparison, (1200, 600)), cv2.COLOR_BGR2RGB))
                else:
                    sample_summary.extend([np.NaN, np.NaN])
                errors.append(sample_summary)
    errors = np.stack(errors)
    summary = pd.DataFrame(errors, index=r_index, columns=['gt_len', 'model_len', 'obstacle_hit'])
    summary.to_csv('summary.csv')
    print(f"Total test samples: {len(dataset)}\nValid samples: {len(errors)}")

                    
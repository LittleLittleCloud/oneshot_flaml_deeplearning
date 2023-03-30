from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
from src.task.task_helper import *
import json
from src.task.config.share_config import *
import os

if __name__ == '__main__':
    device = torch.device('cuda:0')
    result_folder = './result'
    dataset = BLEND_SEARCH_CONFIG["Validate"]
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    for dataset_loader in dataset:
        try:
            dataset, name, num_class = dataset_loader()
            if os.path.exists(f'{result_folder}/{name}.json'):
                print(f'{name} has been run')
                continue
            print(f'run blend search on {name}')
            res = search_config(dataset,
                                name,
                                num_class=num_class,
                                subsample=BLEND_SEARCH_CONFIG["SUB_SAMPLE"],
                                device=device,
                                trail_attempt=BLEND_SEARCH_CONFIG["M"],
                                seed=BLEND_SEARCH_CONFIG["SEED"],
                                epoch=BLEND_SEARCH_CONFIG["EPOCH"])
            print(res)
            with open(f'{result_folder}/{name}.json', 'w') as f:
                json.dump(res, f)
        except Exception as ex:
            print(ex)
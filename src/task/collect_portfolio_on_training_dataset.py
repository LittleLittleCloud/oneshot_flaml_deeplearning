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
    result_folder = './portfolios'
    dataset = COLLECT_PORTFOLIO_CONFIG["Train"]
    dataset.reverse()
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    for dataset_loader in dataset:
        try:
            dataset, name, num_class = dataset_loader()
            for subSample in COLLECT_PORTFOLIO_CONFIG["SUB_SAMPLE"]:
                result_json = f'{result_folder}/{name}-{subSample}.json'
                if os.path.exists(result_json):
                    print(f'{result_json} exists')
                    continue
                print(f'run blend search on {result_json}')
                if subSample == -1:
                    sub_sample = -1
                else:
                    sub_sample = int(len(dataset) * subSample)
                res = search_config(dataset,
                                name,
                                num_class=num_class,
                                subsample=sub_sample,
                                device=device,
                                trail_attempt=COLLECT_PORTFOLIO_CONFIG["M"],
                                seed=COLLECT_PORTFOLIO_CONFIG["SEED"],
                                epoch=COLLECT_PORTFOLIO_CONFIG["EPOCH"])
                print(res)
                with open(result_json, 'w') as f:
                    json.dump(res, f)
        except Exception as ex:
            print(ex)
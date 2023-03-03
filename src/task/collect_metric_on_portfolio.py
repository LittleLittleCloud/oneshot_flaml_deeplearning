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
    dataset = COLLECT_METRIC_CONFIG["Train"]
    metric_folder = COLLECT_METRIC_CONFIG["METRIC_FOLDER"]
    if not os.path.exists(metric_folder):
        os.makedirs(metric_folder)
    portfolio_folder = COLLECT_METRIC_CONFIG["PORTFOLIO_FOLDER"]
    portfolios = os.listdir(portfolio_folder)
    print(f'portfolios: {portfolios}')
    for portfolio_path in portfolios:
        portfolio_full_path = os.path.join(portfolio_folder, portfolio_path)
        portfolio = None
        
        portfolio_name = portfolio_path.split('.')[0]
        portfolio_metric_full_path = os.path.join(metric_folder, portfolio_path)
        metric = {}
        if os.path.exists(portfolio_metric_full_path):
            with open(portfolio_metric_full_path, 'r') as f:
                metric = json.load(f)
        
        with open(portfolio_full_path, 'r') as f:
            _json = json.load(f)
            portfolio = _json['flaml_config'] if 'flaml_config' in _json else {}
            if 'flaml_score' in _json:
                metric[portfolio_name] = _json['flaml_score']['acc']

        print(f'portfolio: {portfolio}') 

        for dataset_loader in dataset:
            try:
                dataset, name, num_class = dataset_loader()
                for subSample in COLLECT_METRIC_CONFIG["SUB_SAMPLE"]:
                    dataset_name = f'{name}-{subSample}'
                    if dataset_name in metric:
                        print(f'{dataset_name} has been completed')
                        continue
                if subSample == -1:
                    sub_sample = -1
                else:
                    sub_sample = int(len(dataset) * subSample)
                
                res = evaluate_config(
                    dataset= dataset,
                    num_class= num_class,
                    config=portfolio,
                    subsample=sub_sample,
                    epoch=COLLECT_METRIC_CONFIG["EPOCH"],
                    seed=COLLECT_METRIC_CONFIG["SEED"],
                    device=device,
                    )
                print(f'evaluate {dataset_name}, score: {res}')
                metric[dataset_name] = res
                with open(portfolio_metric_full_path, 'w') as f:
                    json.dump(metric, f)
            except Exception as ex:
                raise(ex)

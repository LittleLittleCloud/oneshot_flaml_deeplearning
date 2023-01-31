from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
import json

def search_portfolio(dataset_loaders, device):
    for loader in dataset_loaders:
        try:
            dataset, name, num_class = loader()
            search_config(dataset, name, num_class=num_class, subsample=SUB_SAMPLE, device=device, trail_attempt=TRIAL_ATTEMPT, seed=SEED)
        except Exception as ex:
            print(ex)

def evaluate_portfolio(dataset_loaders, device):
    config_dir = 'default/dnn'
    configs = os.listdir(config_dir)
    configs = [x for x in filter(lambda path: path.endswith('json'), configs)]

    result_folder = f'default/result_dnn/'
    configs.append('default.json')
    print(configs)

    for loader in dataset_loaders:
        try:
            dataset,name, num_class = loader()
            result_path = f'{result_folder}{name}.json'
            if os.path.exists(result_path):
                with open(result_path, 'r') as fs:
                    result_obj = json.load(fs)
            else:
                result_obj = {}
            for config in configs:
                if config in result_obj:
                    continue

                if config == 'default.json':
                    config_obj = {}
                else:
                    with open(f'{config_dir}/{config}') as fs:
                        config_obj = json.load(fs)
                
                print(config_obj)
                score = evaluate_config(dataset,
                    num_class=num_class,
                    config=config_obj,
                    subsample=SUB_SAMPLE,
                    device=device,
                    seed=SEED)

                result_obj[config] = {
                    'task': name,
                    'type': 'classification',
                    'result': score,
                    'params': {
                        '_modeljson': config
                    }
                }

                with open(result_path, 'w') as fs:
                    json.dump(result_obj, fs)
        except Exception as ex:
            print(ex, ex.with_traceback)
from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
from src.task.task_helper import *
import json
if __name__ == '__main__':
    device = torch.device('cuda:0')
    validate_json_folder = 'validate/flaml'
    dataset = (DATASET_GROUP_3 + DATASET_GROUP_1)
    #dataset.reverse()
    for dataset_loader in dataset:
        try:
            dataset, name, num_class = dataset_loader()
            if os.path.exists(f'{validate_json_folder}/{name}.json'):
                print(f'{name} has been validated')
                continue
            res = search_config(dataset, name, num_class=num_class, subsample=-1, device=device, trail_attempt=-1, seed=SEED, config_json_folder=validate_json_folder, epoch=5)
            print(res)
            with open(f'{validate_json_folder}/{name}.json', 'w') as f:
                json.dump(res, f)
        except Exception as ex:
            print(ex)

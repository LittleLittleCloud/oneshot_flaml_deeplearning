from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
from src.task.task_helper import *
import json
if __name__ == '__main__':
    device = torch.device('cuda:0')
    validate_json_folder = '/home/azureuser/cloudfiles/code/Users/xiaoyuz/oneshot_flaml_deeplearning/validate/flaml'
    for dataset_loader in DATASET_GROUP_3:
        try:
            dataset, name, num_class = dataset_loader()
            if os.path.exists(f'{validate_json_folder}/{name}.json'):
                print(f'{name} has been validated')
                continue
            res = search_config(dataset, name, num_class=num_class, subsample=-1, device=device, trail_attempt=TRIAL_ATTEMPT, seed=SEED, config_json_folder=validate_json_folder, epoch=20)
            with open(f'{validate_json_folder}/{name}.json', 'w') as f:
                json.dump(res, f)
        except Exception as ex:
            print(ex)

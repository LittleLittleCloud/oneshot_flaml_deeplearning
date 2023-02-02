from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
from src.task.task_helper import *
import json
from flaml.default import suggest_config

def meta_features_fn(dataset_name):
    metaFeaturePath = 'default/metafeatures.json';
    with open(metaFeaturePath, 'r') as f:
        obj = json.load(f)
        meta_feature = obj[dataset_name]
        def fn(*args, **kargs):
            return [meta_feature['number_of_instances'], meta_feature['number_of_features'], meta_feature['number_of_classes']]
        return fn

if __name__ == '__main__':
    device = torch.device('cuda:0')
    validate_json_folder = 'validate/zero_shot'
    for dataset_loader in (DATASET_GROUP_1 + DATASET_GROUP_3):
        try:
            dataset, name, num_class = dataset_loader()
            if os.path.exists(f'{validate_json_folder}/{name}.json'):
                print(f'{name} has been validated')
                continue
            meta_featuresfn = meta_features_fn(name)
            configs = suggest_config('classification', None, None, 'dnn', 'default/output', k=1, meta_feature_fn=meta_featuresfn)
            print(configs[0])
            score = evaluate_config(dataset, num_class, configs[0], -1, 5, device=device)
            obj = {'acc': score, 'config': configs[0]}
            with open(f'{validate_json_folder}/{name}.json', 'w') as f:
                json.dump(obj, f)
        except Exception as ex:
            # raise
            print(ex)

from src.dataset import *
import json
from torch.utils.data import DataLoader, RandomSampler
import torch
import os

meta_feature_path = '/home/azureuser/cloudfiles/code/Users/xiaoyuz/oneshot_flaml_deeplearning/default/metafeatures.json'

dataset_loaders = [
    load_caltech101,
    load_caltech256,
    load_celebA,
    load_cifar10,
    load_country211,
    load_cifar100,
    load_DTD,
    load_EMNIST,
    load_EuroSAT,
    load_FGVCAircraft,
    load_Food101,
    load_FashionMNIST,
    load_Flowers102,
    load_FER2013,
    load_GTSRB,
    load_INaturalist,
    load_KMNIST,
    load_LSUN,
    load_LFWPeople,
    load_OxfordIIITPet,
    load_Omniglot,
    load_Places365,
    load_PCAM,
    load_QMNIST,
]

SEED = 0

rnd = torch.random.manual_seed(SEED)

if os.path.exists(meta_feature_path):
    with open(meta_feature_path, 'r') as f:
        meta_feature = json.load(f)
else:
    meta_feature = {}
for loader in dataset_loaders:
    try:
        dataset, name, num_of_class = loader(transform=transforms.ToTensor())
        if name not in meta_feature:
            print(f'collect meta data for {name}')
            number_of_instance = len(dataset)
            indices = [i for i in range(number_of_instance)]
            dataLoader = DataLoader(dataset=dataset, sampler=RandomSampler(indices, num_samples=1000, generator=rnd))
            # average W * H * C for first 1000 pictures
            features = 0
            for inputs, label in dataLoader:
                (_, c, h, w) = inputs.shape
                features += h * w

            number_of_features = features / 1000
            meta_feature[name] = {
                'number_of_features': number_of_features,
                'number_of_classes': num_of_class,
                'number_of_instances': number_of_instance
            }
            with open(meta_feature_path, 'w') as f:
                json.dump(meta_feature, f)
    except Exception as ex:
        print(ex)
    



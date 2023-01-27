from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
import json
if __name__ == '__main__':
    device = torch.device('cuda:1')
    rnd = torch.random.manual_seed(SEED)
    datasets_loaders = [load_FashionMNIST, load_FER2013, load_FGVCAircraft, load_Flowers102, load_Food101, load_GTSRB, load_INaturalist]

    for loader in datasets_loaders:
        try:
            dataset, name, num_class = loader()
            search_config(dataset, name, num_class=num_class, subsample=SUB_SAMPLE, device=device, trail_attempt=TRIAL_ATTEMPT)
        except Exception as ex:
            print(ex)

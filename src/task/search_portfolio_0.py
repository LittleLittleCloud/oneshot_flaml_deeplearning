from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
import json
if __name__ == '__main__':
    device = torch.device('cuda:0')
    rnd = torch.random.manual_seed(SEED)
    datasets_loaders = [load_DTD, load_country211, load_celebA, load_KMNIST, load_LFWPeople, load_LSUN, load_Omniglot]

    for loader in datasets_loaders:
        try:
            dataset, name, num_class = loader()
            search_config(dataset, name, num_class=num_class, subsample=SUB_SAMPLE, device=torch.device('cuda:0'), trail_attempt=TRIAL_ATTEMPT)
        except Exception as ex:
            print(ex)

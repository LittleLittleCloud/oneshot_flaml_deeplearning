from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
import json
if __name__ == '__main__':
    device = torch.device('cuda:0')
    rnd = torch.random.manual_seed(SEED)
    datasets_loaders = [load_caltech101, load_caltech256, load_cifar10, load_cifar100, load_EMNIST, load_EuroSAT, load_OxfordIIITPet, load_Places365, load_PCAM, load_QMNIST]

    for loader in datasets_loaders:
        try:
            dataset, name, num_class = loader()
            search_config(dataset, name, num_class=num_class, subsample=SUB_SAMPLE, device=device, trail_attempt=TRIAL_ATTEMPT)
        except Exception as ex:
            print(ex)

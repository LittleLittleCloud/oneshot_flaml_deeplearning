from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
from src.task.task_helper import *
import json
if __name__ == '__main__':
    device = torch.device('cuda:1')
    search_portfolio(DATASET_GROUP_2, device)
    evaluate_portfolio(DATASET_GROUP_2, device)

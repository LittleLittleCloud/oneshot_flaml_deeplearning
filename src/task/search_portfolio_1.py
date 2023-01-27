from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
from src.task.task_helper import *
import json
if __name__ == '__main__':
    device = torch.device('cuda:0')
    search_portfolio(DATASET_GROUP_1, device)
    evaluate_portfolio(DATASET_GROUP_1, device)

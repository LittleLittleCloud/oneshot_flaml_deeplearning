from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, random_split

if __name__ == '__main__':
    device = torch.device('cuda:0')
    rnd = torch.random.manual_seed(0)
    datasets_loaders = [load_caltech101, load_caltech201, load_cifar10, load_cifar100]

    for loader in datasets_loaders:
        dataset, name, num_class = loader()
        print(f'search portfolio for {name}')
        train, validate = random_split(dataset, [0.9, 0.1], rnd)
        evaluate_config = run_and_evaluate(train, validate, device, 10, num_class)
        flaml.tune.run(evaluate_config, config_search_space, log_file_name=name)


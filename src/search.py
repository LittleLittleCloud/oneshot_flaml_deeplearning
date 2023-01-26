import flaml
import flaml.tune
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
from src.train import train_and_evaluate_model
from flaml import tune
from src.dataset import load_caltech101
import torch
import time
from src.dataset import *
import json

def run_and_evaluate(train, validate, device, num_epochs, num_classes):
    def evaluate_config(config:dict):
        current_time = time.time()
        metric = train_and_evaluate_model(train, validate, num_classes=num_classes, num_epochs=num_epochs, device=device, **config)
        time2eval = time.time() - current_time
        tune.report(acc=metric, time2eval=time2eval)
    
    return evaluate_config

config_search_space = {
    "batch_size": tune.lograndint(lower = 128 * 5, upper = 128 * 10),
    "learning_rate": tune.loguniform(lower = 1e-5, upper = 1),
    "graident": tune.loguniform(lower = 1e-3, upper = 0.9999),
    "square": tune.loguniform(lower = 1e-3, upper = 0.9999),
    "eps": tune.loguniform(lower = 1e-6, upper = 1e-3),
    "weight_decay": tune.loguniform(lower = 1e-9, upper = 1e-2),
}

def search_config(dataset, name, num_class, subsample = 500, trail_attempt = 30, epoch = 5, seed = 0, device = torch.device('cpu')):
    print(f'search portfolio for {name} dataset len: {len(dataset)}')
    rnd = torch.random.manual_seed(seed)
    indices = [i for i in range(len(dataset))]
    indices = [i for i in RandomSampler(indices, num_samples=subsample, generator=rnd)]
    dataset = Subset(dataset, indices)
    train, validate = random_split(dataset, [0.7, 0.3], rnd)
    evaluate_config = run_and_evaluate(train, validate, device, epoch, num_class)
    # run default
    print('run default')
    automl_default = flaml.tune.run(evaluate_config, {}, log_file_name=name, num_samples = 1, verbose=3, mode='max', metric='acc')
    print('run tune')
    automl = flaml.tune.run(evaluate_config, config_search_space, log_file_name=name, num_samples = trail_attempt, verbose=3, mode='max', metric='acc')
    
    if automl_default.best_result > automl.best_result:
        best_config = automl_default.best_config
    else:
        best_config = automl.best_config
    with open(f'/home/azureuser/cloudfiles/code/Users/xiaoyuz/oneshot_flaml_deeplearning/default/dnn/{name}.json', 'w') as fs:
        json.dump(best_config, fs)

if __name__ == '__main__':
    device = torch.device("cuda:0")
    dataset, dataset_name, num_of_classes = load_caltech101()
    evaluate_config = run_and_evaluate(dataset, dataset, device, 1, num_of_classes)
    flaml.tune.run(evaluate_config, config_search_space, log_file_name=dataset_name)
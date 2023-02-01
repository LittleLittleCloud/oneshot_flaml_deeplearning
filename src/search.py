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
import os

def run_and_evaluate(train, validate, device, num_epochs, num_classes):
    def evaluate_config(config:dict):
        current_time = time.time()
        metric = train_and_evaluate_model(train, validate, num_classes=num_classes, num_epochs=num_epochs, device=device, **config)
        time2eval = time.time() - current_time
        tune.report(acc=metric, time2eval=time2eval)
    
    return evaluate_config

config_search_space = {
    "batch_size": tune.lograndint(lower = 128, upper = 128 * 5),
    "learning_rate": tune.loguniform(lower = 1e-5, upper = 1),
    "graident": tune.loguniform(lower = 1e-3, upper = 0.9999),
    "square": tune.loguniform(lower = 1e-3, upper = 0.9999),
    "eps": tune.loguniform(lower = 1e-6, upper = 1e-3),
    "weight_decay": tune.loguniform(lower = 1e-9, upper = 1e-2),
}

TRIAL_ATTEMPT = 30
SUB_SAMPLE = -1
EPOCH = 5
SEED = 0
CONFIG_JSON_FOLDER = 'default/dnn'
def evaluate_config(dataset,
                num_class,
                config,
                subsample = SUB_SAMPLE,
                epoch = EPOCH,
                seed = SEED,
                device = torch.device('cpu')):
    rnd = torch.random.manual_seed(seed)
    if subsample != -1:
        indices = [i for i in range(len(dataset))]
        indices = [i for i in RandomSampler(indices, num_samples=subsample, generator=rnd)]
        dataset = Subset(dataset, indices)
    train, validate = random_split(dataset, [0.7, 0.3], rnd)
    metric = train_and_evaluate_model(train, validate, num_classes=num_class, num_epochs=epoch, device=device, **config)
    
    return metric


def search_config(
        dataset,
        name,
        num_class,
        subsample = SUB_SAMPLE,
        trail_attempt = TRIAL_ATTEMPT,
        epoch = EPOCH,
        seed = SEED,
        config_json_folder = CONFIG_JSON_FOLDER,
        device = torch.device('cpu')):
    config_json_path = f'{config_json_folder}/{name}.json'
    if os.path.exists(config_json_path):
        print(f'{config_json_path} already exist')
        return

    print(f'search portfolio for {name} dataset len: {len(dataset)}')
    rnd = torch.random.manual_seed(seed)
    train, validate = random_split(dataset, [0.7, 0.3], rnd)
    if subsample != -1:
        indices = [i for i in range(len(train))]
        indices = [i for i in RandomSampler(indices, num_samples=int(subsample * len(train)), generator=rnd)]
        train = Subset(train, indices)
    evaluate_config = run_and_evaluate(train, validate, device, epoch, num_class)
    # run default
    print('run default')
    automl_default = flaml.tune.run(evaluate_config, {}, log_file_name=name, num_samples = 1, verbose=3, mode='max', metric='acc')  
    print(automl_default.best_result)
    
    if trail_attempt > 0:
        print('run tune')
        evaluate_config = run_and_evaluate(train, validate, device, epoch, num_class)
        automl = flaml.tune.run(evaluate_config, config_search_space, log_file_name=name, num_samples = trail_attempt, verbose=3, mode='max', metric='acc')
    else:
        automl = automl_default
    
    if automl_default.best_result['acc'] > automl.best_result['acc']:
        best_config = automl_default.best_config
    else:
        best_config = automl.best_config
    with open(config_json_path, 'w') as fs:
        json.dump(best_config, fs)
    
    return {'default_score': automl_default.best_result, 'flaml_score': automl.best_result, 'flaml_config': automl.best_config}

if __name__ == '__main__':
    device = torch.device("cuda:0")
    dataset, dataset_name, num_of_classes = load_caltech101()
    evaluate_config = run_and_evaluate(dataset, dataset, device, 1, num_of_classes)
    flaml.tune.run(evaluate_config, config_search_space, log_file_name=dataset_name)
import flaml
import flaml.tune
from torch.utils.data import DataLoader, random_split
from train import train_and_evaluate_model
from flaml import tune
from dataset import load_caltech101
import torch

def run_and_evaluate(dataset, device, num_epochs, num_classes):
    train, validate = random_split(dataset, [0.7, 0.3])
    def evaluate_config(config:dict):
        print(config)
        auroc = train_and_evaluate_model(train, validate, num_classes=num_classes, num_epochs=num_epochs, device=device, **config)
        return auroc
    
    return evaluate_config

config_search_space = {
    "batch_size": tune.lograndint(lower = 32, upper = 1024),
    "learning_rate": tune.loguniform(lower = 1e-5, upper = 1),
    "graident": tune.loguniform(lower = 1e-3, upper = 1),
    "square": tune.loguniform(lower = 1e-3, upper = 1),
    "eps": tune.loguniform(lower = 1e-6, upper = 1e-3),
    "weight_decay": tune.loguniform(lower = 1e-9, upper = 1e-2),
}

if __name__ == '__main__':
    device = torch.device("cpu")
    dataset, dataset_name, num_of_classes = load_caltech101()
    evaluate_config = run_and_evaluate(dataset, device, 1, num_of_classes)
    flaml.tune.run(evaluate_config, config_search_space, log_file_name=dataset_name)
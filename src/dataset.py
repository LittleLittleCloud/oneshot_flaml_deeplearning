import tensorflow_datasets as tfds
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision
from torchvision import datasets
import numpy as np

# normalize data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

grayscale_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

rgb_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

def load_caltech101():
    dataset = datasets.Caltech101("data", download=True, transform=grayscale_transform)
    dataset.name = "caltech101"
    return dataset, "caltech101", len(dataset.categories)

def load_caltech201():
    dataset = datasets.Caltech256("data", download=True, transform=grayscale_transform)
    return dataset, "caltech256", len(dataset.categories)

def load_cifar10():
    dataset = datasets.cifar.CIFAR10("data", train=True, download=True, transform=rgb_transform)
    return dataset, "cifar10", len(dataset.class_to_idx)

def load_cifar100():
    dataset = datasets.cifar.CIFAR100("data", train=True, download=True, transform=rgb_transform)
    return dataset, "cifar100", len(dataset.class_to_idx)
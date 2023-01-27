from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from torchvision import datasets
import numpy as np
from torch.utils.data import DataLoader

import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset

def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                print(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

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

# 1
def load_caltech101(transform = grayscale_transform):
    dataset = datasets.Caltech101("data", download=True, transform=transform)
    dataset.name = "caltech101"
    return dataset, "caltech101", len(dataset.categories)
#1
def load_caltech256(transform = grayscale_transform):
    dataset = datasets.Caltech256("data", download=True, transform=transform)
    return dataset, "caltech256", len(dataset.categories)
#1
def load_cifar10(transform = rgb_transform):
    dataset = datasets.cifar.CIFAR10("data", train=True, download=True, transform=transform)
    return dataset, "cifar10", len(dataset.class_to_idx)
#1
def load_cifar100(transform = rgb_transform):
    dataset = datasets.cifar.CIFAR100("data", train=True, download=True, transform=transform)
    return dataset, "cifar100", len(dataset.class_to_idx)
#0
def load_celebA(transform = rgb_transform):
    dataset = datasets.CelebA("data", split="all", download=True, transform=transform)
    return dataset, "celebA", len(dataset.attr_names)
#0
def load_country211(transform = rgb_transform):
    dataset = datasets.Country211("data", split="train", download=True, transform=transform)
    return dataset, "country211", len(dataset.classes)
#0
def load_DTD(transform = rgb_transform):
    dataset = datasets.dtd.DTD("data", split="train", download=True, transform=transform)
    return dataset, "DTD", len(dataset.classes)
#1
def load_EMNIST(transform = grayscale_transform):
    dataset = datasets.EMNIST("data", split="byclass", train=True, download=True, transform=transform)
    return dataset, "EMNIST", len(dataset.classes)
#1
def load_EuroSAT(transform = rgb_transform):
    dataset = datasets.EuroSAT("data", download=True, transform=transform)
    return dataset, "EuroSAT", len(dataset.classes)
#2
def load_FashionMNIST(transform = grayscale_transform):
    dataset = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
    return dataset, "FashionMNIST", len(dataset.classes)
#2
def load_FER2013(transform = grayscale_transform):
    dataset = datasets.FER2013("data", transform=transform)
    return dataset, "FER2013", 7
#2
def load_FGVCAircraft(transform = rgb_transform):
    dataset = datasets.FGVCAircraft("data", download=True, transform=transform, split='train', annotation_level='manufacturer')
    return dataset, "FGVCAircraft", len(dataset.classes)
#2
def load_Flowers102(transform = rgb_transform):
    dataset = datasets.Flowers102("data", download=True, transform=transform, split='train')
    return dataset, "Flowers102", 102
#2
def load_Food101(transform = rgb_transform):
    dataset = datasets.Food101("data", download=True, transform=transform, split='train')
    return dataset, "Food101", 101
#2
def load_GTSRB(transform = rgb_transform):
    dataset = datasets.GTSRB("data", download=True, transform=transform, split='train')
    return dataset, "GTSRB", 43
#2
def load_INaturalist(transform = rgb_transform):
    dataset = datasets.INaturalist("data", download=True, transform=transform)
    return dataset, "INaturalist", len(dataset.all_categories)
#0
def load_KMNIST(transform = grayscale_transform):
    dataset = datasets.KMNIST("data", download=True, transform=transform)
    return dataset, "KMNIST", len(dataset.classes)
#0
def load_LFWPeople(transform = rgb_transform):
    dataset = datasets.LFWPeople("data", download=True, transform=transform)
    return dataset, "LFWPeople", len(dataset.class_to_idx)
#0
def load_LSUN(transform = rgb_transform):
    dataset = datasets.lsun.LSUN("data", transform=transform)
    return dataset, "LSUN", len(dataset.classes)
#0
def load_Omniglot(transform = grayscale_transform):
    dataset = datasets.Omniglot("data", download=True, transform=transform)
    return dataset, "Omniglot", 964

#1
def load_OxfordIIITPet(transform = rgb_transform):
    dataset = datasets.OxfordIIITPet("data", download=True, transform=transform)
    return dataset, "OxfordIIITPet", 37
#1
def load_Places365(transform = rgb_transform):
    dataset = datasets.Places365("data", download=True, transform=transform)
    return dataset, "Places365", len(dataset.classes)
#1
def load_PCAM(transform = rgb_transform):
    dataset = datasets.PCAM("data", download=True, transform=transform)
    return dataset, "PCAM", 2
#1
def load_QMNIST(transform = grayscale_transform):
    dataset = datasets.QMNIST("data", download=True, transform=transform)
    return dataset, "QMNIST", len(dataset.classes)

if __name__ == '__main__':
    dataset, _, _ = load_Omniglot(transform=transforms.ToTensor())
    dataLoader = DataLoader(dataset=dataset)
    labels = {}
    print(len(dataset))
    for _, label in dataLoader:
        labels[label.numpy()[0]] = 0
    print(len(labels))

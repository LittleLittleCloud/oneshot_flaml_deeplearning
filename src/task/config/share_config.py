from src.dataset import *

SHARE_CONFIG = {
    "M": 100, # m step
    "EPOCH": 100,
    "Train": [
        load_DTD,
        load_caltech101,
        load_cifar10,
        load_EMNIST,
        load_EuroSAT,
        load_FGVCAircraft,
        load_GTSRB,
        load_Places365,
        load_StanfordCars,
        load_Food101
    ],
    "Validate": [
        load_SEMEION,
        load_PCAM,
        load_STL10,
        load_caltech256,
        load_cifar100,
        load_celebA,
        load_country211,
        load_FashionMNIST,
        load_Flowers102,
        load_KMNIST,
        load_Omniglot,
        load_OxfordIIITPet,
        load_QMNIST,
        load_RenderedSST2
    ],
    "SUB_SAMPLE": -1, 
    "SEED": 12357
}

BLEND_SEARCH_CONFIG = {
    **SHARE_CONFIG
}

COLLECT_PORTFOLIO_CONFIG = {
    **SHARE_CONFIG,
    "SUB_SAMPLE": [0.1, 0.3, 0.5, 0.7, -1],
    "EPOCH": 30,
    "M": 50
}
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class Model(nn.Module):
    def __init__(self, num_of_class: int, device: torch.device):
        super(Model, self).__init__()
        self.res_net_50 = models.resnet50(pretrained=True)

        # freeze res_net_50
        for param in self.res_net_50.parameters():
            param.requires_grad = False
        
        self.res_net_50.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_of_class)
        )

        self.res_net_50.to(device)

    def forward(self, X):
        return self.res_net_50.forward(X)

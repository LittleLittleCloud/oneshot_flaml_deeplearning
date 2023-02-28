import torch
import torch.nn as nn
from torchvision import models
import numpy as np
class Model(nn.Module):
    
    def __init__(self, num_of_class: int, device: torch.device, arch: str):
        super(Model, self).__init__()
        __archs = {
            'resnet18': models.resnet18(pretrained=True),
            'resnet50': models.resnet50(pretrained=True),
            'resnet101': models.resnet101(pretrained=True),
            'inception': models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT),
            'mobilenet': models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
            }
        
        self.arch = __archs[arch]
        
        __fcs = {
            'resnet18': nn.Sequential(nn.Linear(512, num_of_class), nn.Softmax(dim=0)),
            'resnet50': nn.Sequential(nn.Linear(2048, num_of_class), nn.Softmax(dim=0)),
            'resnet101':nn.Sequential(nn.Linear(2048, num_of_class), nn.Softmax(dim=0)),
            'inception': nn.Sequential(nn.Linear(2048, num_of_class), nn.Softmax(dim=0)),
            'mobilenet': nn.Sequential(nn.Linear(1280, num_of_class), nn.Softmax(dim=0)),
        }

        # freeze arch
        for param in self.arch.parameters():
            param.requires_grad = False
        
        if arch == 'mobilenet':
            self.arch.classifier = __fcs[arch]
        else:
            self.arch.fc = __fcs[arch]

        self.arch = self.arch.to(device)

    def forward(self, X):
        output = self.arch.forward(X)
        if isinstance(output, models.InceptionOutputs):
            return output.logits
        else:
            return output

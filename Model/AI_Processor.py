import torchvision
import torch
import torchaudio
import requests
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu_layer = nn.ReLU()
        self.max_pool_layer = nn.MaxPool2d(kernel_size=2, stirde=2)
        self.fully_conected_layer_1 = nn.Linear(64*8*8, 128)
        self.fully_conected_layer_2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv_layer_1(x))
        x = self.max_pool_layer(x)
        x = F.relu(self.conv_layer_2(x))
        x = self.max_pool_layer(x)
        x = x.view(x.size(), -1)
        x = F.relu(self.fully_conected_layer_1(x))
        x = self.fully_conected_layer_2(x)
        return x

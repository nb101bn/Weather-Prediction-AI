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
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_layer_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu_layer = nn.ReLU()
        self.max_pool_layer = nn.MaxPool2d(kernel_size=2, stirde=2)
        self.final_conv_layer = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv_layer_1(x))
        x = self.max_pool_layer(x)
        x = F.relu(self.conv_layer_2(x))
        x = self.max_pool_layer(x)
        x = F.relu(self.conv_layer_3(x))
        x = self.final_conv_layer(x)
        return x

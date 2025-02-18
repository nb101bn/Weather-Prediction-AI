import torchvision
import torch
import torchaudio
import requests
import pandas as pd
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu_layer = nn.ReLU()
        self.max_pool_layer = nn.MaxPool2d(kernel_size=2, stirde=1)
        self.fully_conected_layer_1 = nn.Linear(7*7*64, 128)
        self.fully_conected_layer_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu_layer(self.conv_layer_1(x))
        x = self.max_pool_layer(x)
        x = self.relu_layer(x)
        x = self.max_pool_layer(x)
        x = x.vew(-1, 7*7*64)
        x = self.relu(self.fully_connected_layer_1(x))
        x = self.fully_connected_layer_2(x)
        return x
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])

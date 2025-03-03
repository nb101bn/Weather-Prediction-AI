import requests
import pandas as pd
import time
import os
import datetime
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np

def ensure_integer(value):
    while not isinstance(value, int):
        if isinstance(value, int):
            return value
        else:
            print('value is not an integer')
            value = int(input())
            return value
'''
def surface_map_requests():
    start_date = datetime.datetime(1985, 1, 1, 0)
    end_date = datetime.datetime(2015, 12, 31, 21)
    def download_png(url, filename):
        response = requests.get(url, stream = True)
        with open(filename, 'wb') as f:
            f.write(response.content)
    while start_date <= end_date:
        year = start_date.strftime('%Y')
        month = start_date.strftime('%m')
        day = start_date.strftime('%d')
        hour = start_date.strftime('%H')
        url = f'https://www.wpc.ncep.noaa.gov/archives/sfc/{year}/sfc{year}{month}{day}{hour}z.gif'
        filename = f'C:\\Users\\natha\Documents\\SurfaceMaps\\Surface_Map_{year}_{month}_{day}_{hour}.png'
        print(url)
        download_png(url, filename)
        start_date += datetime.timedelta(hours=3)
surface_map_requests()
'''
class SurfaceMap_Dataset(Dataset):
    def __init__(self, start_date, end_date, hours_back, transform=None):
        self.start_date = start_date
        self.end_date = end_date
        self.hours_back = hours_back
        self.transform = transform
        plt.ioff()
    def download_png(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            try:
                img = Image.open(BytesIO(response.content))
                return img
            except Image.UnidentifiedImageError:
                print(f'Error unable to identify image at {url}')
                return None
        else:
            print(f'Error: Failed to download image from {url} with status code {response.status_code}')
            return None
    
    def __len__(self):
        return(self.end_date - self.start_date).days +1
    
    def __getitem__(self, idx):
        date = self.start_date + datetime.timedelta(days=idx)
        surface_maps = []
        for i in range(self.hours_back):
            surface_date = date - datetime.timedelta(hours=(i*3))
            year = surface_date.strftime('%Y')
            month = surface_date.strftime('%m')
            day = surface_date.strftime('%d')
            hour = surface_date.strftime('%H')
            surface_url = f'https://www.wpc.ncep.noaa.gov/archives/sfc/{year}/lrgnamsfc{year}{month}{day}{hour}.gif'
            surface_map = self.download_png(surface_url)
            if surface_map is not None:
                surface_map = surface_map.convert('RGB')
                surface_maps.append(surface_map)
        if self.transform:
            surface_maps = [self.transform(sm) for sm in surface_maps]
        if isinstance(surface_maps[0], torch.Tensor):
            surface_maps_tensor = torch.stack(surface_maps, dim=0)
            if surface_maps_tensor.shape[0] == self.hours_back:
                return surface_maps_tensor
            else:
                raise ValueError(f'Expected {self.hours_back} surface maps, but got {surface_maps_tensor.shape[0]}')
        else:
            raise ValueError(f'Expected Tensors but got {type(surface_maps[0])} elements')
        return surface_maps_tensor

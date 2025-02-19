import requests
import pandas as pd
import time
import os
import datetime
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def ensure_integer(value):
    while not isinstance(value, int):
        if isinstance(value, int):
            return value
        else:
            print('value is not an integer')
            value = int(input())
            return value

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

class SurfaceMapDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.lisdir(root_dir) if f.endswith('.png')]
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name =  os.path.join(self.root_dir, self.image_files[idx])
        image = image.open(img_name)
        if self.transform:
            image =self.transform(image)
        return image
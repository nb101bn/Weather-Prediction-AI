import requests
import pandas as pd
import time
import os
import datetime
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def CoCoRaHS_Pull(start_date, end_date):
    def download_png(url, filename):
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
    while start_date <= end_date:
        year = start_date.strftime('%Y')
        month = start_date.strftime('%m')
        day = start_date.strftime('%d')
        url = f'https://www.cocorahs.org/maps/GetMap.aspx?state=usa&type=precip&date={month}/{day}/{year}&cp=0'
        filename = f'C:\\Users\\natha\\Documents\\AITraining\\PrecipObs\\Precip_Obs_24hr_{year}_{month}_{day}.png'
        download_png(url, filename)
        start_date += datetime.timedelta(days=1)
start_date = datetime.datetime(1998, 6, 15)
end_date = datetime.datetime(2015, 12, 31)
CoCoRaHS_Pull(start_date, end_date)
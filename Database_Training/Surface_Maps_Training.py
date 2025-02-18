import requests
import pandas as pd
import time
import os
import datetime

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
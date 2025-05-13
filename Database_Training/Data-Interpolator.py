import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import pandas as pd
import requests
import metpy as mp
import metpy.calc as mpcalc
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import re
from metpy.units import units
import datetime
from scipy.interpolate import griddata

URL = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?data=all&year1=2025&month1=5&day1=1&year2=2025&month2=5&day2=8&network=MO_ASOS&network=IL_ASOS&network=AR_ASOS&network=KY_ASOS&network=TN_ASOS&tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=M&trace=T&direct=no&report_type=3&report_type=4'


def create_dataframe(URL):
    try:
        response = requests.get(URL)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        return df
    except requests.exceptions.RequestException as e:
        print(f'Error fetching data from URL: {e}')
        return None
    except pd.errors.EmptyDataError as e:
        print(f"Error: No data found at URL: {e}")
        return None
    except Exception as e:
        print(f'Error converting to dataframe: {e}')
        return None
def create_map(data):
    domain = [max(data.lat), min(data.lat), max(data.lon), min(data.lon)]
    fig = plt.figure(figsize=(15, 21))
    proj = ccrs.PlateCarree(central_longitude=((domain[2]+domain[3])/2))
    ax = fig.add_subplot(1,1,1, projection=proj)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.OCEAN)
    ax.set_extent([domain[3]-2,domain[2]+2,domain[1]-2,domain[0]+2])
    return ax
def interpolate_to_grid(source, lons, lats, u_wind, v_wind, resolution=30000):  # Resolution in meters
    """
    Interpolates wind components (u, v) to a regular grid.

    Args:
        lons (array-like): Longitudes of the data points.
        lats (array-like): Latitudes of the data points.
        u_wind (array-like): Zonal wind components at the data points.
        v_wind (array-like): Meridional wind components at the data points.
        resolution (int, optional): The grid resolution in meters. Defaults to 3000 (3km).

    Returns:
        tuple: (grid_lon, grid_lat, grid_u, grid_v) -
               Longitudes and latitudes of the grid points, and
               interpolated u and v wind components on the grid.  Returns None on error.
    """
    try:
        # 1. Calculate the bounding box of the data.
        min_lon, max_lon = np.min(source.lon), np.max(source.lon)
        min_lat, max_lat = np.min(source.lat), np.max(source.lat)

        # 2.  Create a regular grid.
        # Convert resolution from meters to degrees (approximate, good for small areas)
        deg_per_km = 1 / 111.0  #  1 degree ≈ 111 km
        grid_res_deg = resolution * deg_per_km / 1000  # km to deg
        grid_lon, grid_lat = np.mgrid[min_lon:max_lon:grid_res_deg, min_lat:max_lat:grid_res_deg]
        # 3. Interpolate the wind components to the grid.
        points = np.column_stack((lons, lats))
        grid_u = griddata(points, u_wind, (grid_lon, grid_lat), method='linear')
        grid_v = griddata(points, v_wind, (grid_lon, grid_lat), method='linear')

        return grid_lon, grid_lat, grid_u, grid_v

    except Exception as e:
        print(f"Error during interpolation: {e}")
        return None
def station_wind(ax, data, date):
    if date is not None:
        try:
            start_date = date
            end_date = date + datetime.timedelta(minutes=59)
            data['valid'] = pd.to_datetime(data['valid'])
            filtered_df = data[(data['valid'] >= start_date) & (data['valid'] <= end_date)]
            filtered_df_unique = filtered_df.drop_duplicates(subset='station', keep='first')
        except Exception as e:
            print(f'Error filtering data set {e}')
            return None

    else:
        print('Date either invalid or non existent')
        return None

    try:
        speed = pd.to_numeric(filtered_df_unique['sknt'], errors='coerce')
        direction = pd.to_numeric(filtered_df_unique['drct'], errors='coerce')
        flat = filtered_df_unique['lat'].values  # Convert to numpy array
        flon = filtered_df_unique['lon'].values # Convert to numpy array
        speed.fillna(0, inplace=True)
        direction.fillna(0, inplace=True)
    except Exception as e:
        print(f'Error parsing dataset {e}')
        return None

    if speed is not None and direction is not None and not speed.empty and not direction.empty:
        try:
            wind_speed = np.array(speed.astype(int)) * units.knots
            wind_direction = np.array(direction.astype(int)) * units.degrees
            u, v = mpcalc.wind_components(wind_speed, wind_direction)
            # Interpolate to grid
            grid_lon, grid_lat, grid_u, grid_v = interpolate_to_grid(data, flon, flat, u, v)
            if grid_lon is not None and grid_lat is not None and grid_u is not None and grid_v is not None:
                # Plot the interpolated wind vectors
                print(f"lon coordinates: {grid_lon}")
                print(f"lat coordinates: {grid_lat}")
                ax.barbs(grid_lon, grid_lat, grid_u, grid_v,
                         transform=ccrs.PlateCarree(), length=5, linewidth=1.4,
                         color='red', zorder=10)  # Use a different color for interpolated winds

            # Plot the station locations and original wind barbs
            #ax.scatter(flon, flat, color='black', transform=ccrs.PlateCarree(), zorder=10)
            #ax.barbs(flon, flat, u, v, transform=ccrs.PlateCarree(), length=5, linewidth=1.4, zorder=10)
            return ax
        except Exception as e:
            print(f"Error plotting or calculating the wind barbs: {e}")
            return None
    else:
        print(f'Dataframe for speed or direction is empty, check dataset')
        return None
df = create_dataframe(URL)
if df is not None:
    print(df.columns)
else:
    print('No data in dataframe')
date = datetime.datetime(2025, 5, 5, 0, 0)
axis = create_map(df)
axis = station_wind(axis, df, date)

plt.show()
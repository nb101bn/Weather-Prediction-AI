import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import io
import pandas as pd
import requests
import metpy.calc as mpcalc
from metpy.units import units
import datetime
from scipy.interpolate import griddata

# URL for fetching meteorological data.  It's good to define constants like this at the top.
DATA_URL = ('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?'
            'data=all&year1=2025&month1=5&day1=1&year2=2025&month2=5&day2=8&'
            'network=MO_ASOS&network=IL_ASOS&network=AR_ASOS&'
            'network=KY_ASOS&network=TN_ASOS&tz=Etc%2FUTC&format=onlycomma&'
            'latlon=yes&elev=yes&missing=M&trace=T&direct=no&'
            'report_type=3&report_type=4')


def create_dataframe(url):
    """
    Fetches data from a given URL and converts it into a Pandas DataFrame.

    Args:
        url (str): The URL to retrieve the data from.

    Returns:
        pandas.DataFrame: A DataFrame containing the fetched data, or None if an error occurs.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        df = pd.read_csv(io.StringIO(response.text))
        return df
    except requests.exceptions.RequestException as e:
        print(f'Error fetching data from URL: {e}')
        return None
    except pd.errors.EmptyDataError as e:
        print(f"Error: No data found at URL: {e}")
        return None
    except Exception as e:
        print(f'Error converting to DataFrame: {e}')
        return None



def create_map(data):
    """
    Creates a Cartopy map with specified features and extent.

    Args:
        data (pandas.DataFrame): DataFrame containing latitude and longitude data.  Assumes
            'lat' and 'lon' columns.

    Returns:
        cartopy.mpl.geoaxes.GeoAxesSubplot: A Cartopy GeoAxesSubplot object representing the map.
    """
    # Calculate the domain from the data.
    domain = [max(data.lat), min(data.lat), max(data.lon), min(data.lon)]

    # Create the figure and axes.
    fig = plt.figure(figsize=(15, 21))
    proj = ccrs.PlateCarree(central_longitude=((domain[2] + domain[3]) / 2))  # Centralize the map
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # Add map features.
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAKES)  # Added missing features.
    ax.add_feature(cfeature.OCEAN)

    # Set the map extent with a small buffer.
    ax.set_extent([domain[3] - 2, domain[2] + 2, domain[1] - 2, domain[0] + 2])
    return ax



def interpolate_to_grid(lons, lats, u_wind, v_wind, resolution=3000):
    """
    Interpolates wind components (u, v) to a regular grid using scipy.interpolate.griddata.

    Args:
        lons (numpy.ndarray): Longitudes of the data points (1D array).
        lats (numpy.ndarray): Latitudes of the data points (1D array).
        u_wind (numpy.ndarray): Zonal wind components at the data points (1D array).
        v_wind (numpy.ndarray): Meridional wind components at the data points (1D array).
        resolution (int, optional): The grid resolution in meters. Defaults to 3000 (3km).

    Returns:
        tuple: (grid_lon, grid_lat, grid_u, grid_v) - Longitudes and latitudes of the grid
            points, and interpolated u and v wind components on the grid. Returns None on error.
    """
    try:
        # 1. Calculate the bounding box of the data.
        min_lon, max_lon = np.min(lons), np.max(lons)
        min_lat, max_lat = np.min(lats), np.max(lats)

        # 2. Create a regular grid.
        deg_per_km = 1 / 111.0  # 1 degree ≈ 111 km
        grid_res_deg = resolution * deg_per_km / 1000  # km to deg
        grid_lon, grid_lat = np.mgrid[min_lon:max_lon:grid_res_deg,
                                   min_lat:max_lat:grid_res_deg]

        # 3. Interpolate the wind components to the grid.
        points = np.column_stack((lons, lats))
        grid_u = griddata(points, u_wind, (grid_lon, grid_lat), method='linear')
        grid_v = griddata(points, v_wind, (grid_lon, grid_lat), method='linear')
        return grid_lon, grid_lat, grid_u, grid_v

    except Exception as e:
        print(f"Error during interpolation: {e}")
        return None



def station_wind(ax, data, date):
    """
    Plots wind barbs at station locations and interpolates wind to a grid.

    Args:
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot): The Cartopy GeoAxesSubplot object to plot on.
        data (pandas.DataFrame): DataFrame containing station data.  Assumes 'station',
            'lat', 'lon', 'sknt', 'drct', and 'valid' columns.
        date (datetime.datetime):  The date for which to plot the wind.
    Returns:
        cartopy.mpl.geoaxes.GeoAxesSubplot: The Cartopy GeoAxesSubplot object with the wind
            barbs plotted, or None on error.
    """
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
        flon = filtered_df_unique['lon'].values  # Convert to numpy array
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
            grid_lon, grid_lat, grid_u, grid_v = interpolate_to_grid(flon, flat, u, v)
            if grid_lon is not None and grid_lat is not None and grid_u is not None and grid_v is not None:
                # Plot the interpolated wind vectors
                ax.barbs(grid_lon, grid_lat, grid_u, grid_v,
                         transform=ccrs.PlateCarree(), length=5, linewidth=1.4,
                         color='red', zorder=10)  # Use a different color

            # Plot the station locations and original wind barbs
            ax.scatter(flon, flat, color='black', transform=ccrs.PlateCarree(), zorder=10)
            ax.barbs(flon, flat, u, v, transform=ccrs.PlateCarree(), length=5, linewidth=1.4, zorder=10)
            return ax
        except Exception as e:
            print(f"Error plotting or calculating the wind barbs: {e}")
            return None
    else:
        print('Dataframe for speed or direction is empty, check dataset')
        return None



def main():
    """
    Main function to orchestrate the data retrieval, processing, and plotting.
    """
    data_frame = create_dataframe(DATA_URL) # Changed from df to data_frame
    if data_frame is not None:
        print(data_frame.columns)
    else:
        print('No data in DataFrame. Exiting.')
        return  # Exit if no data

    analysis_date = datetime.datetime(2025, 5, 5, 0, 0) # Changed from date to analysis_date
    map_axis = create_map(data_frame) # Changed from axis to map_axis
    if map_axis is not None:
        station_wind(map_axis, data_frame, analysis_date)
        plt.show()  # Display the plot
    else:
        print('Error creating map. Exiting.')



if __name__ == "__main__":
    main()  # Call the main function to start the script.
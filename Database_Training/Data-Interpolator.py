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



def wind_interpolate_to_grid(lons, lats, u_wind, v_wind, resolution=30000):
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
            grid_lon, grid_lat, grid_u, grid_v = wind_interpolate_to_grid(flon, flat, u, v)
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

def templike_interpolation(lon, lat, T, resolution=30000):
    """
    Interpolates dewpoint values onto a regular grid.

    Args:
        lon (np.ndarray): 1D array of longitudes of the data points.
        lat (np.ndarray): 1D array of latitudes of the data points.
        dp (np.ndarray): 1D array of dewpoint values corresponding to the lon/lat points.
        resolution (int, optional): Target resolution of the grid in meters. Defaults to 30000.

    Returns:
        tuple (np.ndarray, np.ndarray, np.ndarray):
            - grid_lon: 2D array of longitudes of the interpolated grid.
            - grid_lat: 2D array of latitudes of the interpolated grid.
            - grid_dp: 2D array of interpolated dewpoint values on the grid.
            Returns (None, None, None) if an error occurs during interpolation.
    """
    try:
        # Find the minimum and maximum longitude and latitude to define the grid boundaries.
        min_lon, max_lon = np.min(lon), np.max(lon)
        min_lat, max_lat = np.min(lat), np.max(lat)

        # Calculate the grid resolution in degrees based on the desired resolution in meters.
        deg_per_km = 1 / 111.0  # Approximate degrees per kilometer.
        grid_res_deg = resolution * deg_per_km / 1000  # Convert meters to degrees.

        # Create a 2D regular grid of longitudes and latitudes.
        grid_lon, grid_lat = np.mgrid[min_lon:max_lon:grid_res_deg,
                                        min_lat:max_lat:grid_res_deg]

        # Stack the original longitude and latitude points into a single array.
        points = np.column_stack((lon, lat))

        # Interpolate the dewpoint values onto the new grid using linear interpolation.
        grid_T = griddata(points, T, (grid_lon, grid_lat), method='linear')

        return grid_lon, grid_lat, grid_T

    except Exception as e:
        print(f"Error interpolating the data: {e}")
        return None, None, None

def station_dew_point(ax, data, date):
    """
    Plots station dewpoint values and contours of interpolated dewpoints on a map axis.

    Args:
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot): The map axis to plot on.
        data (pd.DataFrame): DataFrame containing station data with 'valid' (datetime),
                             'station' (str), 'dwpf' (numeric or 'M'), 'lat' (numeric),
                             and 'lon' (numeric) columns.
        date (datetime.datetime): The specific date and time for which to plot the data.

    Returns:
        cartopy.mpl.geoaxes.GeoAxesSubplot: The modified map axis. Returns None if an error occurs
                                             during station value plotting.
    """
    if date is not None:
        try:
            # Define the start and end times for filtering the data (one-hour window).
            start_date = date
            end_date = date + datetime.timedelta(minutes=59)

            # Ensure the 'valid' column is in datetime format.
            data['valid'] = pd.to_datetime(data['valid'])

            # Filter the DataFrame to include data within the specified time window.
            filtered_df = data[(data['valid'] >= start_date) & (data['valid'] <= end_date)]

            # Remove duplicate stations, keeping the first observation.
            filtered_df_unique = filtered_df.drop_duplicates(subset='station', keep='first')

        except Exception as e:
            print(f"Error filtering the dataframe: {e}")
            return ax
    else:
        print("date is empty please enter a proper date.")
        return ax

    try:
        # Extract dewpoint, latitude, and longitude values from the filtered DataFrame.
        dewpoint = pd.to_numeric(filtered_df_unique['dwpf'], errors='coerce')
        flat = filtered_df_unique['lat'].values
        flon = filtered_df_unique['lon'].values

        # Fill any NaN values in the dewpoint series with 0.
        dewpoint.fillna(0, inplace=True)

        # Replace any 'M' (missing) values in the dewpoint series with 0.
        dewpoint = dewpoint.replace('M', 0)

        # Ensure dewpoint is a NumPy array.
        dewpoint.values

    except Exception as e:
        print(f"Error parsing values: {e}")
        return ax

    # Interpolate dewpoint values onto a regular grid.
    grid_lon, grid_lat, dews = templike_interpolation(flon, flat, dewpoint)

    if grid_lon is not None and grid_lat is not None and dews is not None:
        try:
            # Plot the interpolated dewpoint values as text on the map.
            for lon_idx, lat_idx in np.ndindex(dews.shape):
                value = dews[lon_idx, lat_idx]
                lon = grid_lon[lon_idx, lat_idx]
                lat = grid_lat[lon_idx, lat_idx]
                if not np.isnan(value):
                    ax.text(lon, lat, f"{value:.1f}",
                            color="blue",
                            transform=ccrs.PlateCarree(),
                            fontsize=6,
                            ha='center',
                            va='center',
                            zorder=5)
        except Exception as e:
            print(f'Error plotting the interpolated dewpoints: {e}')

        try:
            # Ensure the interpolated dewpoints array is of float type for contouring.
            dews = np.array(dews, dtype=float)

            # Calculate the minimum and maximum interpolated dewpoint values, ignoring NaNs.
            min_dews = np.nanmin(dews)
            max_dews = np.nanmax(dews)

            # Generate contour levels every 5 units within the range of the data.
            contour_levels = np.arange(np.floor(min_dews/5)*5, np.ceil(max_dews/5)*5+1, 5)
            print(f"Contour Levels: {contour_levels}") # For debugging purposes

            # Plot the dewpoint contours on the map.
            contour = ax.contour(grid_lon, grid_lat, dews, levels=contour_levels,
                                colors='black', linewidths=1, transform=ccrs.PlateCarree())

            # Add labels to the contour lines.
            ax.clabel(contour, inline=True, fontsize=5, fmt="%1.0f")

        except Exception as e:
            print(f"Error plotting contours for interpolated dewpoints: {e}")
    else:
        print(f"Either Grid_lon, Grid_lat, or Dews is missing or nan")

    try:
        # Plot the actual dewpoint values from the stations as text on the map.
        for lon, lat, value in zip(flon, flat, dewpoint):
            if not np.isnan(value):
                ax.text(lon, lat,
                        f"{value:.1f}",
                        color='green',
                        transform=ccrs.PlateCarree(),
                        fontsize=6,
                        ha='center',
                        va='center',
                        zorder=5)
            else:
                print(f"Value is either nan or empty please input an actual value")
                return None # Consider if you want to return here or continue plotting other stations
    except Exception as e:
        print(f"Error plotting dewpoint {e}")
        return None

    return ax

def station_temperature(ax, data, date):
    """
    Plots station temperature values and contours of interpolated temperatures on a map axis.

    Args:
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot): The map axis to plot on.
        data (pd.DataFrame): DataFrame containing station data with 'valid' (datetime),
                             'station' (str), 'tmpf' (numeric or 'M'), 'lat' (numeric),
                             and 'lon' (numeric) columns.
        date (datetime.datetime): The specific date and time for which to plot the data.

    Returns:
        cartopy.mpl.geoaxes.GeoAxesSubplot: The modified map axis. Returns None if an error occurs
                                             during station value plotting.
    """
    if date is not None:
        try:
            # Define the start and end times for filtering the data (one-hour window).
            start_date = date
            end_date = date + datetime.timedelta(minutes=59)

            # Ensure the 'valid' column is in datetime format.
            data['valid'] = pd.to_datetime(data['valid'])

            # Filter the DataFrame to include data within the specified time window.
            filtered_df = data[(data['valid'] >= start_date) & (data['valid'] <= end_date)]

            # Remove duplicate stations, keeping the first observation.
            filtered_df_unique = filtered_df.drop_duplicates(subset='station', keep='first')

        except Exception as e:
            print(f"Error filtering the dataframe: {e}")
            return ax
    else:
        print("Date is either nan or empty")
        return ax

    try:
        # Extract temperature, longitude, and latitude values from the filtered DataFrame.
        temp = pd.to_numeric(filtered_df_unique['tmpf'], errors='coerce')
        flon = filtered_df_unique['lon']
        flat = filtered_df_unique['lat']

        # Fill any NaN values in the temperature series with 0.
        temp.fillna(0, inplace=True)

        # Replace any 'M' (missing) values in the temperature series with 0.
        temp = temp.replace('M', 0)

        # Ensure temperature is a NumPy array.
        temp.values

    except Exception as e:
        print(f"Error gathering temperature, lon, and lat values from Filtered dataset: \n {e}")
        return ax

    # Interpolate temperature values onto a regular grid.
    grid_lon, grid_lat, grid_temp = templike_interpolation(flon, flat, temp)

    if grid_lon is not None and grid_lat is not None and grid_temp is not None:
        try:
            # Plot the interpolated temperature values as text on the map.
            for lon_idx, lat_idx in np.ndindex(grid_temp.shape):
                value = grid_temp[lon_idx, lat_idx]
                lon = grid_lon[lon_idx, lat_idx]
                lat = grid_lat[lon_idx, lat_idx]
                if not np.isnan(value):
                    ax.text(lon, lat, f"{value:.1f}",
                            color="red" if value >= 33 else "blue",
                            transform=ccrs.PlateCarree(),
                            fontsize=6,
                            ha='center',
                            va='center',
                            zorder=5)
        except Exception as e:
            print(f'Error plotting the interpolated temperatures: {e}')

        try:
            # Ensure the interpolated temperature array is of float type for contouring.
            grid_temp = np.array(grid_temp, dtype=float)

            # Calculate the minimum and maximum interpolated temperature values, ignoring NaNs.
            min_temp = np.nanmin(grid_temp)
            max_temp = np.nanmax(grid_temp)

            # Generate contour levels every 5 units within the range of the data.
            contour_levels = np.arange(np.floor(min_temp/5)*5, np.ceil(max_temp/5)*5+1, 5)
            print(f"Contour Levels: {contour_levels}") # For debugging purposes

            # Plot the temperature contours on the map.
            contour = ax.contour(grid_lon, grid_lat, grid_temp, levels=contour_levels,
                                colors='black', linewidths=1, transform=ccrs.PlateCarree())

            # Add labels to the contour lines.
            ax.clabel(contour, inline=True, fontsize=5, fmt="%1.0f")

        except Exception as e:
            print(f"Error plotting contours for interpolated temperatures: {e}")

    try:
        # Plot the actual temperature values from the stations as text on the map.
        for lon, lat, value in zip(flon, flat, temp):
            if not np.isnan(value):
                ax.text(lon, lat, f"{value:.1f}",
                        color='red' if value >= 33 else 'blue',
                        transform=ccrs.PlateCarree(),
                        fontsize=6,
                        ha='center',
                        va='center',
                        zorder=5)
            else:
                print('Error plotting temperature values, either lon, lat, or value is nan or empty')
                return None # Consider if you want to return here or continue plotting other stations
    except Exception as e:
        print(f'Error converting or plotting temperature: {e}')
        return None

    return ax

def station_pressure(ax, data, date):
    if date is not None:
        try:
            start_date = date
            end_date = date+ datetime.timedelta(minutes=59)
            data['valid'] = pd.to_datetime(data['valid'])
            filtered_df = data[(data['valid']>=start_date)&(data['valid']<=end_date)]
            filtered_df_unique = filtered_df.drop_duplicates(subset='station', keep='first')
        except Exception as e:
            print(f"Error filtering data: {e} \n please try again.")
            return ax
    else:
        print(f"Error date is either empty or nan please enter a valid date.")
        return ax
    try:
        pressure = pd.to_numeric(filtered_df_unique['mslp'], errors='coerce')
        flon = filtered_df_unique['lon'].values
        flat = filtered_df_unique['lat'].values
        pressure.fillna(1000, inplace=True)
        pressure = pressure.replace('M', 1000)
        pressure.values
    except Exception as e:
        print(f"Error gathering variables {e}")
        return ax
    # Interpolate pressure data
    grid_lon, grid_lat, grid_pressure = templike_interpolation(flon, flat, pressure, resolution=3000)

    if grid_lon is not None and grid_lat is not None and grid_pressure is not None:
        try:
            min_pressure = np.nanmin(grid_pressure)
            max_pressure = np.nanmax(grid_pressure)
            contour_levels = np.arange(np.floor(min_pressure / 6) * 6, np.ceil(max_pressure / 6) * 6 + 1, 6) # Adjust interval as needed
            contour = ax.contour(grid_lon, grid_lat, grid_pressure, levels=contour_levels,
                                colors='black', linewidths=1, transform=ccrs.PlateCarree())
            ax.clabel(contour, inline=True, fontsize=5, fmt="%1.0f")
        except Exception as e:
            print(f"Error contouring pressure lines: {e}")
            return None
    else:
        print("Error during pressure interpolation, cannot contour.")
        return None

    return ax


def main():
    
    analysis_date = datetime.datetime(2024, 5, 5, 0, 0) # Changed from date to analysis_date
    
    # URL for fetching meteorological data.  It's good to define constants like this at the top.
    DATA_URL = ('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?'
                f'data=all&year1={analysis_date.strftime('%Y')}&month1={analysis_date.strftime('%m')}&day1={analysis_date.strftime('%d')}'
                f'&year2={analysis_date.strftime('%Y')}&month2={analysis_date.strftime('%m')}&day2={analysis_date.strftime('%d')}&'
                'network=MO_ASOS&network=IA_ASOS&tz=Etc%2FUTC&format=onlycomma&'
                'latlon=yes&elev=yes&missing=M&trace=T&direct=no&'
                'report_type=3&report_type=4')
    
    """
    Main function to orchestrate the data retrieval, processing, and plotting.
    """
    data_frame = create_dataframe(DATA_URL) # Changed from df to data_frame
    if data_frame is not None:
        print(data_frame.columns)
    else:
        print('No data in DataFrame. Exiting.')
        return  # Exit if no data
    map_axis = create_map(data_frame) # Changed from axis to map_axis
    if map_axis is not None:
        station_pressure(map_axis, data_frame, analysis_date)
        plt.show()  # Display the plot
    else:
        print('Error creating map. Exiting.')



if __name__ == "__main__":
    main()  # Call the main function to start the script.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.interpolate import griddata

def extract_basement_tiff_data(input_raster):
    with rasterio.open(input_raster) as src:
        transform = src.transform  # Affine transformation
        data = src.read(1)  # Read raster data
        nodata_value = src.nodata  # NoData value
        crs_utm = src.crs  # Get the CRS (should be UTM)

        # Transformer to convert from UTM (EPSG:32614) to WGS 84 (EPSG:4326)
        transformer = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

        # Get raster dimensions
        height, width = data.shape

        # Create lists to store extracted values
        latitudes, longitudes, elevations = [], [], []

        # Loop over each pixel in the raster
        for row in range(height):
            for col in range(width):
                # Convert pixel coordinates to UTM
                utm_x, utm_y = transform * (col, row)

                # Convert UTM to Latitude/Longitude
                lon, lat = transformer.transform(utm_x, utm_y)

                # Get the elevation value
                elevation = data[row, col]

                # Skip NoData values
                if elevation == nodata_value:
                    continue

                # Append to lists
                latitudes.append(lat)
                longitudes.append(lon)
                elevations.append(elevation)

    # Create a DataFrame
    df = pd.DataFrame({"Latitude": latitudes, "Longitude": longitudes, "Elevation": elevations})
    return df

def get_basement_cross_plot_data(input_raster,start_point,end_point,width_in_deg):
    start_lat, start_lon = start_point
    end_lat, end_lon = end_point
    width_deg = width_in_deg
    
    df = extract_basement_tiff_data(input_raster)
    df = df[(df["Latitude"] >= start_lat-width_deg) & (df["Latitude"] <= end_lat+width_deg)]
    df = df[(df["Longitude"] >= start_lon-width_deg) & (df["Longitude"] <= end_lon+width_deg)]

    # Extract lat, lon, and elevation values from DataFrame
    lats = df["Latitude"].values
    lons = df["Longitude"].values
    elevations = df["Elevation"].values

    # Generate evenly spaced longitudes between start and end
    num_points = 500  # Increase for higher resolution
    lon_profile = np.linspace(start_lon, end_lon, num_points)

    # Interpolate latitude and elevation for these longitudes
    lat_profile = np.interp(lon_profile, lons, lats)  # Interpolating latitudes
    elev_profile = griddata((lons, lats), elevations, (lon_profile, lat_profile), method="linear")

    cross_plot_data = pd.DataFrame({"Longitude": lon_profile, "Latitude": lat_profile, "Elevation": elev_profile})
    return cross_plot_data

def plot_ts_tp_multiple_stations(df, stations, ax=None, xlim=None,ylim=None,
                                 time_col='origin_time', ts_tp_col='ts-tp',
                                 colors=None,
                                 plot_seismicity=False,
                                 plot_error_bar=False,
                                 plot_iqr=True,window='90D',title=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df[df['station'].isin(stations)]

    if not colors:
      palette = sns.color_palette("tab10", len(stations))

    else:
      palette = colors
    station_colors = dict(zip(stations, palette))

    # Define region colors
    #unique_regions = sorted(df[region_col].unique())
    #palette = sns.color_palette("tab10", len(unique_regions))
    #region_colors = dict(zip(unique_regions, palette))

    for station in stations:
        station_df = df[df['station'] == station]
        if plot_seismicity:
          ax.scatter(station_df[time_col], station_df[ts_tp_col],
                   label=station, s=10, alpha=0.7,
                   color=station_colors[station], edgecolors='k', linewidths=0.2)
        # IQR shading per station
        if plot_iqr and len(station_df) >= 10:
            station_df = station_df.set_index(time_col).sort_index()
            rolling = station_df[ts_tp_col].rolling(window)

            q25 = rolling.quantile(0.25)
            q75 = rolling.quantile(0.75)
            station_df.reset_index(inplace=True)
            ax.fill_between(
                q25.index,
                q25,
                q75,
                color=station_colors[station],
                alpha=0.2,
                label=f"{station} IQR"
            )
        if plot_error_bar:
          station_df = station_df.set_index(time_col).sort_index()
          rolling = station_df[ts_tp_col].rolling(window)
          q25 = rolling.quantile(0.25)
          q50 = rolling.quantile(0.5)
          q75 = rolling.quantile(0.75)

          station_df.reset_index(inplace=True)

          ax.errorbar(
                station_df[time_col],
                station_df[ts_tp_col],
                yerr=[q50-q25,q75-q50],
                fmt='o',
                markersize=2,
                label=station,
                color=station_colors[station],
                alpha=0.4,
                capsize=2,
                elinewidth=1,
                linewidth=0.5
            )
    if title:
      ax.set_title("ts - tp vs Time by Station")
    ax.set_xlabel("Time")
    ax.set_ylabel("ts - tp (s)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    if xlim:
        ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))
    if ylim:
        ax.set_ylim(ylim)

    ax.legend(title="Station", bbox_to_anchor=(1.01, 1), loc='upper left')

    return ax

def plot_depth_vs_time_by_region_with_iqr(
    df,
    output_path=None,
    basement_min=None,
    basement_max=None,
    xlim=None,
    ylim=None,
    dpi=300,
    colors=None,
    time_col='origin_time',
    depth_col='z_new_from_surface',
    region_col='region',
    basement_col='basement_elevation_from_sea_level',
    only_regions=["R1", "R2", "R3"],
    plot_seismicity=False,
    ax=None,
    legend=True,
    title=False
):
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # Drop NaNs
    df = df.dropna(subset=[depth_col, region_col, basement_col])

    if only_regions:
        df = df[df[region_col].isin(only_regions)]

    # Filter by basement elevation
    if basement_min is not None:
        df = df[df[basement_col] >= basement_min]
    if basement_max is not None:
        df = df[df[basement_col] <= basement_max]

    # Create figure/axes if not provided
    external_ax = ax is not None
    if not external_ax:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    unique_regions = sorted(df[region_col].unique())
    # Define region colors
    if not colors:
      palette = sns.color_palette("tab10", len(unique_regions))
    else:
      palette = colors
    region_colors = dict(zip(unique_regions, palette))

    # Plot each region
    for region in unique_regions:
        region_df = df[df[region_col] == region].sort_values(by=time_col)

        if plot_seismicity:
            ax.scatter(
                region_df[time_col],
                region_df[depth_col],
                s=10,
                alpha=0.5,
                label=region,
                color=region_colors[region],
                edgecolors='k',
                linewidths=0.2
            )

        if len(region_df) >= 10:
            rolling = region_df.set_index(time_col)[depth_col].rolling('90D')
            q25 = rolling.quantile(0.25)
            q75 = rolling.quantile(0.75)

            ax.fill_between(
                q25.index,
                q25,
                q75,
                color=region_colors[region],
                alpha=0.2,
                label=f"{region} IQR"
            )
            ax.plot(q25.index, q25, color=region_colors[region], linewidth=2, linestyle='--', alpha=1)
            ax.plot(q75.index, q75, color=region_colors[region], linewidth=2, linestyle='--', alpha=1)

    # Axis formatting
    if title:
      ax.set_title("Depth vs Time by Region with IQR Shading")
    ax.set_xlabel("Time")
    ax.set_ylabel("Depth from Surface (km)")


    if xlim:
        ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))
    if ylim:
        ax.set_ylim(ylim)
    if legend:
      ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    ax.invert_yaxis()
    plt.tight_layout()

    # Save if path is provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')

    return fig, ax

def plot_depth_vs_time(
    df,
    output_path,
    xlim=None,
    ylim=None,
    dpi=300,
    time_col='origin_time',
    depth_col='z_new_from_surface',
    region_col='region'
):
    # Convert time column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create color palette
    regions = sorted(df[region_col].dropna().unique())
    palette = sns.color_palette("tab10", len(regions))
    region_colors = dict(zip(regions, palette))

    # Plot by region
    for region in regions:
        subset = df[df[region_col] == region]
        ax.scatter(
            subset[time_col],
            subset[depth_col],
            s=10,
            alpha=0.7,
            label=region,
            color=region_colors[region],
            edgecolors='k',
            linewidths=0.2
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Depth from Surface (km)")
    ax.set_title("Earthquake Depth from Surface vs Time by Region")
    ax.invert_yaxis()  # Depth increases downward

    # Apply limits if provided
    if xlim:
        ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))
    if ylim:
        ax.set_ylim(ylim)

    ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    #plt.close(fig)

    return fig,ax

if __name__ == "__main__":
    # Example usage:
    plot_depth_vs_time(df_events, "depth_vs_time.png", xlim=["2018-01-01", "2024-12-31"], ylim=[4, 13])
    plt.show()
    
    plot_depth_vs_time_by_region_with_iqr(df_events, "depth_vs_time_by_basement.png",
                                      xlim=["2018-01-01", "2024-06-01"],
                                      ylim=[4, 14],
                                basement_min=2.0,
    basement_max=6.0,)
    plt.show()
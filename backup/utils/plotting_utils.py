import copy
import numpy as np

import matplotlib.pyplot as plt
import folium
import geopandas as gpd
from shapely.wkt import loads
from mpl_toolkits.axes_grid1 import make_axes_locatable



def convert_df_to_geo_df(data):
    geo_data = copy.copy(data)
    geo_data['geom'] = geo_data['geom'].apply(lambda x: loads(x))
    geo_data = gpd.GeoDataFrame(geo_data, geometry='geom')
    geo_data.crs = {'init': 'epsg:4326'}
    return geo_data


def plot_surface(df):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    df.plot(column='prediction', ax=ax, legend=True, cax=cax, cmap='YlOrRd', linewidth=0.1, edgecolor='black')
    plt.show()


def plot_time_series_comparison(series1, series2, label1, label2, times):
    plt.figure(figsize=(15, 6))
    plt.plot(times, series1, marker='.', color='r', label=label1)
    plt.plot(times, series2, marker='.', color='olive', label=label2)
    plt.xlabel('Time')
    plt.ylabel('Air Quality Values')
    plt.legend(prop={'size': 15})
    plt.show()


def plot_base_map(coord_df, zoom_start=11):
    min_lat, max_lat = min(coord_df['lat']), max(coord_df['lat'])
    min_lon, max_lon = min(coord_df['lon']), max(coord_df['lon'])
    start_lat, start_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
    m = folium.Map(location=[start_lat, start_lon], zoom_start=zoom_start)
    return m


def plot_numeric_geo_data(data, m, gid, feature):
    folium.Choropleth(
        data,
        data=data,
        columns=[gid, feature],
        key_on='feature.properties.{}'.format(gid),
        fill_color='BuPu',
        fill_opacity=0.7,
        line_opacity=0.5,

    ).add_to(m)
    return m


def plot_marker(data, lon, lat, m, color):
    folium.Marker(
        location=[lat, lon],
        tooltip=data,
        icon=folium.Icon(color=color),

    ).add_to(m)
    return m

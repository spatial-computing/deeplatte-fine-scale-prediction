import copy
import numpy as np
import matplotlib.pyplot as plt
import folium
import geopandas as gpd
from shapely.wkt import loads
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import contextily as ctx


def convert_df_to_geo_df(data):
    geo_data = copy.copy(data)
    geo_data['geom'] = geo_data['geom'].apply(lambda x: loads(x))
    geo_data = gpd.GeoDataFrame(geo_data, geometry='geom')
    geo_data.crs = {'init': 'epsg:4326'}
    return geo_data


def plot_surface(tar_col, df, **kwargs):
    
    df = convert_df_to_geo_df(df)
    df = df.to_crs(epsg=3857)
    v_min = min(df[tar_col]) if kwargs.get('v_min') is None else kwargs['v_min']
    v_max = max(df[tar_col]) if kwargs.get('v_max') is None else kwargs['v_max']
    figsize = (10, 10) if kwargs.get('figsize') is None else kwargs['figsize']
    alpha = 0.7 if kwargs.get('alpha') is None else kwargs['alpha']
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    df.plot(column=tar_col, ax=ax, legend=True, cax=cax, cmap=matplotlib.cm.get_cmap('RdYlGn_r'), 
            vmin=v_min, vmax=v_max, linewidth=0.1, edgecolor='black', alpha=alpha)
    
    """ some optional configuration """
    if kwargs.get('basemap') is not None and kwargs['basemap']:
        ax.axis('off')
        ctx.add_basemap(ax, zoom=12)
    if kwargs.get('title') is not None:
        ax.set_title(kwargs['title'])
    if kwargs.get('xlabel') is not None:
        ax.xlabel(kwargs['xlabel'])
    if kwargs.get('ylabel') is not None:
        ax.ylabel(kwargs['ylabel'])
    plt.show()
    plt.close()

    
def plot_multiple_surfaces(tar_col, df_list, r, c, **kwargs):
    
    fig = plt.figure(figsize=(c*6, r*8))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    for i in range(len(df_list)):
        
        ax = fig.add_subplot(r, c, i + 1)
        df = df_list[i]
        df = convert_df_to_geo_df(df).to_crs(epsg=3857)
        v_min = min(df[tar_col]) if kwargs.get('v_min') is None else kwargs['v_min']
        v_max = max(df[tar_col]) if kwargs.get('v_max') is None else kwargs['v_max']
        alpha = 0.7 if kwargs.get('alpha') is None else kwargs['alpha']
    
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        df.plot(column=tar_col, ax=ax, legend=True, cax=cax, cmap=matplotlib.cm.get_cmap('RdYlGn_r'), 
                vmin=v_min, vmax=v_max, linewidth=0.1, edgecolor='black', alpha=0.7)

        """ some optional configuration """
        if kwargs.get('basemap') is not None and kwargs['basemap']:
            ax.axis('off')
            ctx.add_basemap(ax, zoom=12)
        if kwargs.get('titles') is not None:
            ax.set_title(kwargs['titles'][i])
        if kwargs.get('xlabel') is not None:
            ax.xlabel(kwargs['xlabel'])
        if kwargs.get('ylabel') is not None:
            ax.ylabel(kwargs['ylabel'])
    plt.show()
    plt.close()


def plot_multiple_time_series(series_list, label_list, x_ticks, **kwargs):
    
    cmap = plt.cm.get_cmap("hsv", len(label_list) + 1)
    figsize = (15, 6) if kwargs.get('figsize') is None else kwargs['figsize']

    plt.figure(figsize=figsize)
    for i, _ in enumerate(series_list):
        plt.plot(x_ticks, series_list[i], marker='.', color=cmap(i), label=label_list[i])

    """ some optional configuration """
    if kwargs.get('title') is not None:
        plt.title(kwargs['title'])
    if kwargs.get('xlabel') is not None:
        plt.xlabel(kwargs['xlabel'])
    else:
        plt.xlabel('Time')
    if kwargs.get('ylabel') is not None:
        plt.ylabel(kwargs['ylabel'])
    else:
        plt.ylabel('Air Quality Values')
    plt.legend(prop={'size': 15})
    plt.show()


def plot_basemap(coord_df, zoom_start=11):
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

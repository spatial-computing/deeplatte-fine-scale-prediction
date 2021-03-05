import sys

# sys.path.append('/home/eva/jonsnow_air_quality')
from prisms_data_preprocessing.conn_postgresql.common_db import Base, session, engine, meta
from prisms_data_preprocessing.gen_mapping_mat import *
from prisms_data_preprocessing.utils import mapcity

import pandas as pd
import pytz
import numpy as np
from sqlalchemy import Table, extract
from datetime import datetime,timedelta
import os.path


def gen_grid_data(n_times, n_features, geo_dict, mapping_mat):
    n_rows, n_cols = mapping_mat.shape
    tar_data = np.full([n_times, n_features, n_rows, n_cols], np.nan)
    for i in range(n_rows):
        for j in range(n_cols):
            gid = mapping_mat[i][j]
            try:
                idx = geo_dict[gid]
                if idx is not None:
                    tar_data[..., i, j] = idx
            except:
                continue
    return tar_data


def gen_grid_data_old(ori_data, loc_list, mapping_mat):
    """
    transferring original data to the matrix data

    params:
        ori_data: (n_times, n_features, n_loc)
        loc_list: list of corresponding locations
    return:
        tar_data: (n_times, n_features, n_rows, n_cols)
    """
    n_rows, n_cols = mapping_mat.shape
    n_times, n_features, n_loc = ori_data.shape  # n_times = 1 means transforming to a 2D array

    tar_data = np.full([n_times, n_features, n_rows, n_cols], np.nan)

    def get_index(tar):
        try:
            return loc_list.index(tar)
        except ValueError:
            return None

    for i in range(n_rows):
        for j in range(n_cols):
            gid = mapping_mat[i][j]
            idx = get_index(gid)
            if idx is not None:
                tar_data[..., i, j] = ori_data[..., idx]
    return tar_data


def gen_label_mat(pm_obj, city_id, timelist, mapping_mat):
    """
    construct the label matrix, if there is no label for a grid, using Nan to fill in.

    return:
        pm_mat: (n_times, n_output=1, n_rows, n_cols)
    """
    pm_query_sql = session.query(pm_obj.c.ogc_fid, pm_obj.c.timestamp, pm_obj.c.percentile_cont) \
        .filter(pm_obj.c.timestamp >= timelist[0]) \
        .filter(pm_obj.c.timestamp <= timelist[-1]) \
        .filter(pm_obj.c.city_id == city_id)

    pm_data = pd.read_sql(pm_query_sql.statement, session.bind)

    pm_mat_list = []
    for t in timelist:
        this_pm_data = pm_data[pm_data['timestamp'] == t]
        this_pm_dict = dict(this_pm_data[['ogc_fid', 'percentile_cont']].values)
        # this_pm_grids = list(this_pm_data['ogc_fid'])
        # this_pm_data = np.array(this_pm_data['percentile_cont']).reshape((1, 1, -1))
        this_pm_mat = gen_grid_data(1, 1, this_pm_dict, mapping_mat)
        pm_mat_list.append(this_pm_mat)

    pm_mat = np.vstack(pm_mat_list)
    print('The shape of PM matrix = {}.'.format(pm_mat.shape))
    return pm_mat


def gen_meo_vector(clima_obj, city_id, time_list, mapping_mat):
    """
    load weather data and construct the meo vector
    return:
        meo_vector: (n_times, n_loc, n_meo_features)
    """
    min_time, max_time = time_list[0], time_list[-1]
    meo_data = session.query(clima_obj.c.gid, clima_obj.c.timestamp, clima_obj.c.data) \
        .filter(clima_obj.c.timestamp >= min_time) \
        .filter(clima_obj.c.timestamp <= max_time)
    meo_data = pd.read_sql(meo_data.statement, session.bind)
    meo_list = []
    for t in time_list:
        this_data = meo_data[meo_data['timestamp'] == t]
        this_dict = dict(this_data[['gid', 'data']].values)
        # this_pm_grids = list(this_pm_data['ogc_fid'])
        # this_pm_data = np.array(this_pm_data['percentile_cont']).reshape((1, 1, -1))
        this_mat = gen_grid_data(1, len(this_data["data"].iloc[0]), this_dict, mapping_mat)
        meo_list.append(this_mat)
    meo_mat = np.vstack(meo_list)
    print('The shape of PM matrix = {}.'.format(meo_mat.shape))
    return meo_mat


def gen_geo_vector(geo_obj, geo_name_obj, grid_list):
    """
    load geographic data and construct the geographic vector

    return:
        geo_vector: (n_loc, n_geo_features)
        geo_name_list: a list of geographic feature names
    """

    geo_data = session.query(geo_obj.c.ogc_fid, geo_obj.c.data) \
        .filter(geo_obj.c.ogc_fid.in_(grid_list)) \
        .order_by(geo_obj.c.ogc_fid).all()

    n_geo_features = len(geo_data[0][1])
    geo_dict = dict(geo_data)

    # print('The shape of geographic vector = {}.'.format(geo_vector.shape))

    geo_name_df = pd.read_sql(session.query(geo_name_obj).statement, session.bind)
    geo_name_list = list(geo_name_df['name'])

    if len(geo_name_list) != n_geo_features:
        print('Something wrong with the geographic feature vector!')

    return geo_dict, geo_name_list


def get_gridobj(tablename, schema='geographic_data'):
    return Table(tablename, meta, autoload=True, autoload_with=engine, schema=schema)


def mainbymonth(years, months, city_ids):
    mapping_mat = np.load(f'data/Los_Angeles_1000m_grid_mat.npz')['mat']
    grid_list = set(mapping_mat.flatten().tolist())
    output = dict()
    output["purple_air"] = {}
    output["climacell"] = {}
    # static data (geo vector)
    geo_obj = get_gridobj(f'{"los_angeles"}_{1000}m_grid_geo_vector', "preprocess")
    geo_name_obj = get_gridobj(f'{"los_angeles"}_{1000}m_grid_geo_name', "preprocess")

    geo_dict, geo_name_list = gen_geo_vector(geo_obj, geo_name_obj, grid_list)
    static_mat = gen_grid_data(1, len(geo_name_list), geo_dict, mapping_mat)
    output["static_mat"] = static_mat
    output["static_features"] = geo_name_list

    for year in years:
        pm_obj = get_gridobj(f'prod_1000m_grid_purple_air_pm25_{year}', "preprocess")
        pm_obj = session.query(pm_obj).filter(pm_obj.c.ogc_fid.in_(grid_list)).subquery()
        for month in months:
            for city_id in city_ids:
                climacell_obj = get_gridobj(
                    f'prod_1000m_grid_meo_climacell_interpolate_{year}{str(month).rjust(2, "0")}', "preprocess")
                climacell_obj = session.query(climacell_obj).filter(climacell_obj.c.gid.in_(grid_list)).subquery()
                min_time = f'{year}-{month}-01-04'
                max_time = f'{year + (month + 1) // 12}-{month + 1 if (month + 1) % 12 != 0 else 1}-01-04'
                tz = pytz.timezone('America/Los_Angeles')  # should change according to city timezone
                time_list = pd.date_range(start=min_time, end=max_time, closed='left', freq='1H')
                time_list = sorted(list(set([tz.localize(x) for x in time_list])))

                climacell_mat = gen_meo_vector(climacell_obj, city_id, time_list, mapping_mat)
                pm_mat = gen_label_mat(pm_obj, city_id, time_list, mapping_mat)

                output["purple_air"][f'{year}{str(month).rjust(2, "0")}'] = pm_mat
                output["climacell"][f'{year}{str(month).rjust(2, "0")}'] = climacell_mat

                print(f'done {year}{str(month).rjust(2, "0")}')

    return output


def generateTimeSection(min_time, max_time):
    tz = pytz.timezone('America/Los_Angeles')
    time_sections = []
    next = datetime(min_time.year, min_time.month, 1, 4)
    if next <= min_time:
        next = datetime(min_time.year + min_time.month // 12, min_time.month % 12 + 1, 1, 4)
    while next < max_time:
        time_list = pd.date_range(start=min_time, end=next, closed='left', freq='1H')
        time_list = sorted(list(set([tz.localize(x) for x in time_list])))
        time_sections.append(time_list)
        min_time = next
        next = datetime(next.year + next.month // 12, next.month % 12 + 1, 1, 4)
    time_list = pd.date_range(start=min_time, end=max_time, closed='left', freq='1H')
    time_list = sorted(list(set([tz.localize(x) for x in time_list])))
    time_sections.append(time_list)
    return time_sections


def check_mapping_mat_exist(filename, res, city, city_id):
    if os.path.isfile(filename):
        print(f'{filename} exists.')
        return
    else:
        print(f'{filename} does not exist. Generate now ...')
        gen_mapping_mat(res, city, city_id)
        print(f'{filename} exists.')


def gen_train_data(min_time, max_time, res=1000, city="Los Angeles"):
    if city not in mapcity:
        exit("Please input correct city name")
    city_id = mapcity[city]
    cityname = "_".join(city.split()).lower()

    max_time = datetime.strptime(max_time,"%Y-%m-%d-%H")
    min_time = datetime.strptime(min_time, "%Y-%m-%d-%H")

    check_mapping_mat_exist(f'prisms_data_preprocessing/data/{cityname}_{res}m_grid_mat.npz', res, cityname, city_id)
    mapping_mat = np.load(f'prisms_data_preprocessing/data/{cityname}_{res}m_grid_mat.npz')['mat']
    grid_list = set(mapping_mat.flatten().tolist())
    output = dict()
    output["label_mat"] = []
    output["dynamic_mat"] = []
    # static data (geo vector)
    geo_obj = get_gridobj(f'{cityname.lower()}_{res}m_grid_geo_vector', "preprocess")
    geo_name_obj = get_gridobj(f'{cityname.lower()}_{res}m_grid_geo_name', "preprocess")

    geo_dict, geo_name_list = gen_geo_vector(geo_obj, geo_name_obj, grid_list)
    static_mat = gen_grid_data(1, len(geo_name_list), geo_dict, mapping_mat)
    output["static_mat"] = static_mat
    output["static_features"] = geo_name_list

    time_sections = generateTimeSection(min_time, max_time)

    prevYear = None
    for ts in time_sections:
        month = (ts[-1]+timedelta(hours=-4)).month
        year = (ts[-1]+timedelta(hours=-4)).year

        if year != prevYear:
            if year == 2020:
                pm_obj = get_gridobj(f'prod_{res}m_grid_purple_air_pm25_{year}_09_12', "preprocess")
            else:
                pm_obj = get_gridobj(f'prod_{res}m_grid_purple_air_pm25_{year}', "preprocess")
            pm_obj = session.query(pm_obj).filter(pm_obj.c.ogc_fid.in_(grid_list)).subquery()
        climacell_obj = get_gridobj(f'prod_{res}m_grid_meo_climacell_interpolate_{year}{str(month).rjust(2, "0")}', \
                                    "preprocess")
        climacell_obj = session.query(climacell_obj).filter(climacell_obj.c.gid.in_(grid_list)).subquery()
        climacell_mat = gen_meo_vector(climacell_obj, city_id, ts, mapping_mat)
        pm_mat = gen_label_mat(pm_obj, city_id, ts, mapping_mat)

        if len(output["label_mat"]) > 0:
            output["label_mat"] = np.vstack((output["label_mat"], pm_mat))
            output["dynamic_mat"] = np.vstack((output["dynamic_mat"], climacell_mat))
        else:
            output["label_mat"] = pm_mat
            output["dynamic_mat"] = climacell_mat

        prevYear = year
        print(f'starttime: {ts[0]}  endtime:{ts[-1]} done')

    output["mapping_mat"] = mapping_mat
    output["dynamic_features"] = ["summary", 'precip_intensity', 'temperature', 'dew_point', 'humidity', 'pressure',
                                  'wind_speed', 'wind_bearing', 'cloud_cover', 'visibility']

    return output


if __name__ == "__main__":
    print("hello you are importing gen_train_data.py")
    a = gen_train_data('2020-12-1-3','2021-2-15-4')
    min_time = datetime(2021, 1, 1, 4)
    max_time = datetime(2021, 1, 2, 4)
    city_id = 2
    mapping_mat = np.load(f'data/Los_Angeles_1000m_grid_mat.npz')['mat']
    grid_list = set(mapping_mat.flatten().tolist())
    output = dict()
    output["purple_air"] = {}
    output["climacell"] = {}
    # static data (geo vector)
    geo_obj = get_gridobj(f'{"los_angeles"}_{1000}m_grid_geo_vector', "preprocess")
    geo_name_obj = get_gridobj(f'{"los_angeles"}_{1000}m_grid_geo_name', "preprocess")

    geo_dict, geo_name_list = gen_geo_vector(geo_obj, geo_name_obj, grid_list)
    static_mat = gen_grid_data(1, len(geo_name_list), geo_dict, mapping_mat)
    output["static_mat"] = static_mat
    output["static_features"] = geo_name_list

    tz = pytz.timezone('America/Los_Angeles')  # should change according to city timezone
    time_list = pd.date_range(start=min_time, end=max_time, closed='left', freq='1H')
    time_list = sorted(list(set([tz.localize(x) for x in time_list])))

    year = 2021
    pm_obj = get_gridobj(f'prod_1000m_grid_purple_air_pm25_{year}', "preprocess")
    pm_obj = session.query(pm_obj).filter(pm_obj.c.ogc_fid.in_(grid_list)).subquery()
    month = 1
    city_id = 2
    climacell_obj = get_gridobj(
        f'prod_1000m_grid_meo_climacell_interpolate_{year}{str(month).rjust(2, "0")}', "preprocess")
    climacell_obj = session.query(climacell_obj).filter(climacell_obj.c.gid.in_(grid_list)).subquery()

    for year in [2021]:
        pm_obj = get_gridobj(f'prod_1000m_grid_purple_air_pm25_{year}', "preprocess")
        for month in [1]:
            for city_id in [2]:
                climacell_obj = get_gridobj(
                    f'prod_1000m_grid_meo_climacell_interpolate_{year}{str(month).rjust(2, "0")}', "preprocess")

                min_time = f'{year}-{month}-01-04'
                max_time = f'{year + (month + 1) // 12}-{month + 1 if (month + 1) % 12 != 0 else 1}-01-04'
                tz = pytz.timezone('America/Los_Angeles')  # should change according to city timezone
                time_list = pd.date_range(start=min_time, end=max_time, closed='left', freq='1H')
                time_list = sorted(list(set([tz.localize(x) for x in time_list])))

                climacell_mat = gen_meo_vector(climacell_obj, city_id, time_list, mapping_mat)
                pm_mat = gen_label_mat(pm_obj, city_id, time_list, mapping_mat)

                output["purple_air"][f'{year}{str(month).rjust(2, "0")}'] = pm_mat
                output["climacell"][f'{year}{str(month).rjust(2, "0")}'] = climacell_mat

                print(f'done {year}{str(month).rjust(2, "0")}')

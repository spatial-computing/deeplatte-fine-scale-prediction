import sys
sys.path.append('/home/eva/jonsnow_air_quality')

from conn_postgresql.common_db import Base, session, engine,meta

import pandas as pd
import pytz
import numpy as np
from sqlalchemy import Table,extract

def gen_grid_data(ori_data, loc_list, mapping_mat):
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
        this_pm_grids = list(this_pm_data['ogc_fid'])
        this_pm_data = np.array(this_pm_data['percentile_cont']).reshape((1, 1, -1))
        this_pm_mat = gen_grid_data(this_pm_data, this_pm_grids, mapping_mat)
        pm_mat_list.append(this_pm_mat)

    pm_mat = np.vstack(pm_mat_list)
    print('The shape of PM matrix = {}.'.format(pm_mat.shape))
    return pm_mat

def gen_meo_vector(clima_obj, city_id, time_list,grid_list):
    """
    load weather data and construct the meo vector

    return:
        meo_vector: (n_times, n_loc, n_meo_features)
    """
    min_time, max_time = time_list[0], time_list[-1]
    n_times, n_loc = len(time_list), len(grid_list)

    meo_data = session.query(clima_obj.c.data) \
        .filter(clima_obj.c.timestamp >= min_time) \
        .filter(clima_obj.c.timestamp <= max_time) \
        .filter(clima_obj.c.gid.in_(grid_list)) \
        .order_by(clima_obj.c.timestamp, clima_obj.c.gid).all()

    n_meo_features = len(meo_data[0][0])

    meo_vector = np.array(meo_data).reshape((n_times, n_loc, n_meo_features))

    print('The shape of meo vector = {}.'.format(meo_vector.shape))
    return meo_vector

def get_gridobj( tablename, schema='geographic_data'):
    return Table(tablename, meta, autoload=True, autoload_with=engine, schema=schema)

def main(year,month):
    mapping_mat = np.load(f'data/Los_Angeles_1000m_grid_mat.npz')['mat']
    output = dict()
    output["purple_air"] = {}
    output["climacell"] = {}
    for year in year:
        pm_obj = get_gridobj(f'prod_1000m_grid_purple_air_pm25_{year}', "preprocess")
        for month in month:
            for city_id in [2]:
                climacell_obj = get_gridobj(
                    f'prod_1000m_grid_meo_climacell_interpolate_{year}{str(month).rjust(2, "0")}', "preprocess")

                grid_list = set(mapping_mat.flatten().tolist())

                min_time = f'{year}-{month}-01-04'
                max_time = f'{year+(month+1)//12}-{month +1 if month%12 != 0 else 1}-01-04'
                tz = pytz.timezone('America/Los_Angeles') # should change according to city timezone
                time_list = pd.date_range(start=min_time, end=max_time, closed='left', freq='1H')
                time_list = sorted(list(set([tz.localize(x) for x in time_list])))

                pm_mat = gen_label_mat(pm_obj, 2, time_list, mapping_mat)
                climacell_meo = gen_meo_vector(climacell_obj, city_id, time_list, grid_list)

                output["purple_air"][f'{year}{str(month).rjust(2,"0")}'] = pm_mat
                output["climacell"][f'{year}{str(month).rjust(2, "0")}'] = climacell_meo

                print(f'done {year}{str(month).rjust(2,"0")}')
    return output
if __name__ == "__main__":
    mapping_mat = np.load(f'data/Los_Angeles_1000m_grid_mat.npz')['mat']
    output = dict()
    output["purple_air"] = {}
    output["climacell"] = {}
    for year in [2021]:
        pm_obj = get_gridobj(f'prod_1000m_grid_purple_air_pm25_{year}', "preprocess")
        for month in [1,2,3,4,5]:
            for city_id in [2]:
                climacell_obj = get_gridobj(
                    f'prod_1000m_grid_meo_climacell_interpolate_{year}{str(month).rjust(2, "0")}', "preprocess")

                grid_list = set(mapping_mat.flatten().tolist())

                min_time = f'{year}-{month}-01-04'
                max_time = f'{year+(month+1)//12}-{month +1 if (month+1)%12 != 0 else 1}-01-04'
                tz = pytz.timezone('America/Los_Angeles') # should change according to city timezone
                time_list = pd.date_range(start=min_time, end=max_time, closed='left', freq='1H')
                time_list = sorted(list(set([tz.localize(x) for x in time_list])))

                pm_mat = gen_label_mat(pm_obj, 2, time_list, mapping_mat)
                climacell_meo = gen_meo_vector(climacell_obj, city_id, time_list, grid_list)

                output["purple_air"][f'{year}{str(month).rjust(2,"0")}'] = pm_mat
                output["climacell"][f'{year}{str(month).rjust(2, "0")}'] = climacell_meo

                print(f'done {year}{str(month).rjust(2,"0")}')

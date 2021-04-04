from data_preprocessing.db_conn.common_db import Base, session, engine, meta

import pandas as pd
import numpy as np
from sqlalchemy import Table


def endPoints(coord_df):
    colList = coord_df["easting"].drop_duplicates().sort_values()
    rowList = coord_df["northing"].drop_duplicates().sort_values()
    easting = colList.iloc[len(colList) // 2]
    northing = rowList.iloc[len(rowList) // 2]
    rows = coord_df["easting"].value_counts()[easting]
    cols = coord_df["northing"].value_counts()[northing]
    i = 0
    diff = 0
    while (coord_df["easting"] == colList.iloc[i]).sum() - diff != rows:
        if i >= 3:
            i = -1
            diff += 1
        i += 1
    minX = colList.iloc[i]
    j = 0
    diff = 0
    while (coord_df["northing"] == rowList.iloc[j]).sum() - diff != cols:
        if j >= 3:
            j = -1
            diff += 1
        j += 1
    minY = rowList.iloc[j]
    i = -1
    diff = 0
    while (coord_df["easting"] == colList.iloc[i]).sum() - diff != rows:
        if i <= -3:
            i = 0
            diff += 1
        i -= 1
    maxX = colList.iloc[i]
    j = -1
    diff = 0
    while (coord_df["northing"] == rowList.iloc[j]).sum() - diff != cols:
        print(j, rowList.iloc[j])
        if j <= -3:
            j = 0
            diff += 1
        j -= 1
    maxY = rowList.iloc[j]

    return rows, cols, minX, maxX, minY, maxY


def gen_matrix(coord_obj, city_id=2):
    coord_df = pd.read_sql(session.query(coord_obj).filter(coord_obj.c.city_id == city_id).statement, session.bind)
    coord_df["easting"] = coord_df["easting"].apply(lambda x: int(x[:-2]))
    coord_df["northing"] = coord_df["northing"].apply(lambda x: int(x[:-2]))

    n_rows, n_cols, minX, maxX, minY, maxY = endPoints(coord_df)
    mat = np.full(([n_rows, n_cols]), -1)
    for i in range(len(coord_df)):
        if coord_df.iloc[i]["easting"] < minX or coord_df.iloc[i]["easting"] > maxX or coord_df.iloc[i][
            "northing"] < minY or coord_df.iloc[i]["northing"] > maxY:
            continue
        c = (coord_df.iloc[i]["easting"] - minX) // 1000
        r = n_rows - 1 - (coord_df.iloc[i]["northing"] - minY) // 1000
        mat[r][c] = coord_df.iloc[i]["ogc_fid"]
    return mat


def get_gridobj(res, schema='geographic_data'):
    return Table('grid_{}m'.format(res), meta, autoload=True, autoload_with=engine, schema=schema)


# def gettableobj(resolution):
#     class grid_1000m(GridTemplate, Base):
#         __tablename__ = 'grid_{}m'.format(resolution)
#     return grid_1000m

def gen_mapping_mat(res=1000, city="los_angeles", city_id=2):
    coordobj = get_gridobj(res)

    mat = gen_matrix(coordobj, city_id)
    global_n_rows, global_n_cols = mat.shape
    output_file = f'data_preprocessing/data/{city}_{res}m_grid_mat.npz'
    print(output_file)
    print('Number of rows = {}.'.format(global_n_rows))
    print('Number of cols = {}.'.format(global_n_cols))

    np.savez_compressed(
        output_file,
        mat=mat
    )


if __name__ == "__main__":
    coordobj = get_gridobj(1000)
    mat = gen_matrix(coordobj, 2)
    global_n_rows, global_n_cols = mat.shape
    output_file = 'data/los_angeles_1000m_grid_mat.npz'
    print('Number of rows = {}.'.format(global_n_rows))
    print('Number of cols = {}.'.format(global_n_cols))

    np.savez_compressed(
        output_file,
        mat=mat
    )

import math

import json
import os

import copy
import pandas as pd


from scripts.data_loader import DataObj, load_train_val_test
from scripts.result_viz import spatial_viz, temporal_viz, output_prediction
from utils.metrics import *


def physical_dis(coord1, coord2):

    lat1, lon1 = coord1
    lat2, lon2 = coord2
    radius = 6371  # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return radius * c


def main(args, **kwargs):

    """ load data object """
    data = np.load(kwargs['data_file'])
    data_obj = DataObj(data['label_mat'], None, None, None, None, None,
                       data['dynamic_features'], data['static_features'], data['mapping_mat'])
    geom_df = pd.read_csv(kwargs['geom_file'])
    grids = list(geom_df['gid'])

    """ load train, val, test locations """
    f = open(kwargs['train_val_test_file'], 'r')
    train_val_test = json.loads(f.read())

    data_obj.train_loc = train_val_test[args.month]['train_loc']
    data_obj.val_loc = train_val_test[args.month]['val_loc']
    data_obj.test_loc = train_val_test[args.month]['test_loc']

    data_obj.train_y = data_obj.gen_train_val_test_label(data_obj.label_mat, data_obj.train_loc)
    data_obj.val_y = data_obj.gen_train_val_test_label(data_obj.label_mat, data_obj.val_loc)
    data_obj.test_y = data_obj.gen_train_val_test_label(data_obj.label_mat, data_obj.test_loc)

    """ transfer geom_df to a 2-D array """
    geom_arr = np.array(geom_df[['lon', 'lat']])
    geom_arr = geom_arr.T.reshape(1, 2, -1)

    def get_index(tar):
        try:
            return grids.index(tar)
        except ValueError:
            return None

    geom_mat = np.full([1, 2, data_obj.n_rows, data_obj.n_cols], np.nan)  # (1, 2, 92, 76) / (1, 2, 46, 38)
    for i in range(data_obj.n_rows):
        for j in range(data_obj.n_cols):
            gid = data_obj.mapping_mat[i][j]
            idx = get_index(gid)
            if idx is not None:
                geom_mat[..., i, j] = geom_arr[..., idx]

    """ generate weight matrix """
    weight_mat = np.full([len(data_obj.train_loc), data_obj.n_rows, data_obj.n_cols], 0.0)
    for i_loc, loc in enumerate(data_obj.train_loc):
        this_coord = geom_df[geom_df['gid'] == loc][['lon', 'lat']].values[0]
        for i in range(data_obj.n_rows):
            for j in range(data_obj.n_cols):
                coord = geom_mat[0, :, i, j]
                dis = physical_dis(this_coord, coord)

                if 0.0 <= dis: 
                    weight_mat[i_loc, i, j] = 1.0 / dis if dis != 0 else 4.0
                    # weight_mat[i_loc, i, j] = np.exp(-dis)

    """ make prediction using IDW """
    # find the row number and column number for each train location
    train_loc_rc = []
    for i_loc, loc in enumerate(data_obj.train_loc):
        r, c = np.where(data_obj.mapping_mat == loc)
        train_loc_rc.append((r[0], c[0]))

    predictions = []
    for t in range(data_obj.n_times):
        this_pm_data = np.array([data_obj.train_y[t, 0, r, c] for (r, c) in train_loc_rc])
        this_pm_data = np.expand_dims(this_pm_data, axis=-1)
        this_pm_data = np.expand_dims(this_pm_data, axis=-1)
        this_pm_data = np.repeat(this_pm_data, data_obj.n_rows, axis=1)
        this_pm_data = np.repeat(this_pm_data, data_obj.n_cols, axis=2)

        weighted_sum = np.nansum(weight_mat * this_pm_data, axis=0)

        mask = copy.copy(this_pm_data)
        mask[~np.isnan(mask)] = 1.0
        sum_weight = np.nansum(weight_mat * mask, axis=0)
        predictions.append(weighted_sum / sum_weight)

    prediction = np.stack(predictions)

    """ evaluation """
    prediction = prediction.reshape(data_obj.n_times, 1, data_obj.n_rows, data_obj.n_cols)[args.seq_len:, ...]
    rmse, mape, r2 = compute_error(data_obj.test_y[args.seq_len:, ...], prediction)
    print(f'# {args.month} : RMSE = {rmse:.4f}, MAPE = {mape:.4f}, R2 = {r2:.4f}')

    # """ draw the spatial prediction """
    mean_prediction = np.nanmean(prediction, axis=0)
    spatial_viz(mean_prediction, data_obj, args, **kwargs)

    # """ draw the temporal predictions """
    # loc = data_obj.test_loc[3]
    # temporal_viz(prediction, loc, data_obj, args)

    # """ Write the output """
    # output_prediction(prediction, data_obj, param, **kwargs)


class Param:
    def __init__(self, year):
        """ default configuration """
        self.area = 'los_angeles'
        self.resolution = 500
        self.year = year
        self.month = '01'
        self.seq_len = 6


if __name__ == '__main__':

    # define directory
    year = 2020
    param = Param(year)
    data_dir = f'/Users/yijunlin/Research/PRISMS2020/training_data/'
    result_dir = '/Users/yijunlin/Research/PRISMS2020/results/IDW/'
    train_val_test_file = os.path.join(data_dir, f'train_val_test_los_angeles_500m_{year}_sub_region_in_sample_4321.json')
    geom_file = os.path.join(data_dir, '{}_{}m_grid.csv'.format(param.area, param.resolution))
    kwargs = dict()

    for month in range(1, 4):

        month_str = str(month).rjust(2, '0')
        next_month_str = str(month % 12 + 1).rjust(2, '0')
        next_year = year if month != 12 else year + 1
        param.month = month_str
        data_file = os.path.join(data_dir + f'los_angeles_500m_{year}',
                                 '{}_{}m_{}_{}.npz'.format(param.area, param.resolution, param.year, param.month))
        kwargs['data_dir'] = data_dir
        kwargs['result_dir'] = result_dir
        kwargs['geom_file'] = geom_file
        kwargs['data_file'] = data_file
        kwargs['train_val_test_file'] = train_val_test_file
        kwargs['min_time'] = f'{year}-{month_str}-01'
        kwargs['max_time'] = f'{next_year}-{next_month_str}-01'

        main(param, **kwargs)

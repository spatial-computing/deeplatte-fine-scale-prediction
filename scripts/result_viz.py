import os
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.plotting_utils import *


def spatial_viz(prediction, data_obj, args, **kwargs):
    """ draw the spatial prediction """

    mapping_mat_copy = data_obj.mapping_mat.reshape(1, data_obj.n_rows, data_obj.n_cols)
    prediction_with_location = np.vstack([prediction, mapping_mat_copy]).reshape(2, -1)

    prediction_with_location = np.moveaxis(prediction_with_location, -1, 0)
    prediction_with_location = pd.DataFrame(prediction_with_location, columns=['prediction', 'gid'])
    prediction_with_location['gid'] = prediction_with_location['gid'].astype(int)

    geom_df = pd.read_csv(os.path.join(kwargs['data_dir'], '{}_{}m_grid.csv'.format(args.area, args.resolution)))
    geom_df = prediction_with_location.merge(geom_df, on='gid')
    predition_geom_df = convert_df_to_geo_df(geom_df)
    plot_surface(predition_geom_df)


def temporal_viz(prediction, loc, data_obj, args):
    """ draw the temporal prediction """

    print('Testing locations: {}'.format(loc))
    loc_index_r, loc_index_c = np.where(data_obj.mapping_mat == loc)
    plot_time_series_comparison(data_obj.test_y[args.seq_len:, 0, loc_index_r[0], loc_index_c[0]].reshape(-1),
                                prediction[:, 0, loc_index_r[0], loc_index_c[0]].reshape(-1),
                                'Ground Truth', 'Prediction',
                                [i for i in range(prediction.shape[0])])


def output_prediction(prediction, data_obj, args, **kwargs):
    mapping_mat_copy = data_obj.mapping_mat.reshape(1, 1, data_obj.n_rows, data_obj.n_cols)
    mapping_mat_copy = np.repeat(mapping_mat_copy, prediction.shape[0], axis=0)
    mapping_mat_copy = mapping_mat_copy.reshape(-1)
    mapping_mat_copy_df = pd.DataFrame(mapping_mat_copy, columns=['gid'])

    # get time list
    min_time, max_time = kwargs['min_time'], kwargs['max_time']
    tz = pytz.timezone('America/Los_Angeles')
    time_list = pd.date_range(start=min_time, end=max_time, closed='left', freq='1H')
    time_list = sorted(list(set([tz.localize(x) for x in time_list])))
    time_array = np.array(time_list[args.seq_len:]).reshape(-1, 1, 1, 1)
    time_array = np.repeat(time_array, repeats=data_obj.n_rows, axis=2)
    time_array = np.repeat(time_array, repeats=data_obj.n_cols, axis=3)
    time_array = time_array.reshape(-1)
    time_array_df = pd.DataFrame(time_array, columns=['timestamp'])

    predictions_df = pd.DataFrame(prediction.reshape(-1), columns=['prediction'])
    labels_df = pd.DataFrame(data_obj.label_mat[args.seq_len:, ...].reshape(-1), columns=['label'])

    output_df = pd.concat([mapping_mat_copy_df, time_array_df, predictions_df, labels_df], axis=1)
    output_df['gid'] = output_df['gid'].astype(int)

    output_path = os.path.join(kwargs['result_dir'], '{}_{}m_{}_{}_prediction.csv'
                               .format(args.area, args.resolution, args.year, args.month))
    output_df.to_csv(output_path, index=False)

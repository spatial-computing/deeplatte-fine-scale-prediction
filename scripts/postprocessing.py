import os
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymongo

from utils.plotting_utils import *


def extract_selected_features(dapm, features, mask_thre):
    """ print out the selected features """

    params = [p for p in dapm.mask_layer.linear_layer.parameters()]

    weights = []
    weight = params[0].data.cpu().numpy()
    for i in range(weight.shape[0]):
        weights.append(weight[i, i])

    feature_weight = {}
    for i, w in enumerate(weights):
        if abs(w) >= mask_thre:
            feature_weight[features[i]] = abs(w)

    sorted_feature_weights = [[k, v] for k, v in sorted(feature_weight.items(), key=lambda item: -item[1])]
    return sorted_feature_weights


def spatial_viz(prediction, mapping_mat, geom_df, **kwargs):
    """ draw the spatial prediction """

    n_rows, n_cols = mapping_mat.shape
    mapping_mat_copy = mapping_mat.reshape(1, n_rows, n_cols)
    prediction_with_location = np.vstack([prediction, mapping_mat_copy]).reshape(2, -1)

    prediction_with_location = np.moveaxis(prediction_with_location, -1, 0)
    prediction_with_location = pd.DataFrame(prediction_with_location, columns=['prediction', 'gid'])
    prediction_with_location['gid'] = prediction_with_location['gid'].astype(int)

    geom_df = prediction_with_location.merge(geom_df, on='gid')
    geom_df = geom_df.dropna()
    if len(geom_df) > 0:
        plot_surface('prediction', geom_df, **kwargs)


def temporal_viz(prediction, tar_locations, label_mat, mapping_mat, **kwargs):
    """ draw the temporal prediction """

    for loc in tar_locations:
        print('Testing locations: {}'.format(loc))
        loc_index_r, loc_index_c = np.where(mapping_mat == loc)
        series_list = [label_mat[:, 0, loc_index_r[0], loc_index_c[0]].reshape(-1), prediction[:, 0, loc_index_r[0], loc_index_c[0]].reshape(-1)]
        label_list = ['Ground Truth', 'Prediction']
        x_ticks = [i for i in range(prediction.shape[0])]
        plot_multiple_time_series(series_list, label_list, x_ticks, **kwargs)
            
            
def write_prediction_csv(prediction, label_mat, mapping_mat, args, **kwargs):
    """ write the predictions to a file """
    
    n_times, _, n_rows, n_cols = prediction.shape
    mapping_mat_copy = mapping_mat.reshape(1, 1, n_rows, n_cols)
    mapping_mat_copy = np.repeat(mapping_mat_copy, n_times, axis=0)
    
    # get time list
    min_time = args.min_time
    max_time = args.max_time
    tz = pytz.timezone('America/Los_Angeles')
    time_list = pd.date_range(start=min_time, end=max_time, closed='left', freq='1H')
    time_list = sorted(list(set([tz.localize(x) for x in time_list])))
    time_arr = np.array(time_list[-n_times:]).reshape(-1, 1, 1, 1)
    time_arr = np.repeat(time_arr, repeats=n_rows, axis=2)
    time_arr = np.repeat(time_arr, repeats=n_cols, axis=3)
    
    output = np.concatenate([mapping_mat_copy, time_arr, prediction, label_mat], axis=1)
    output = np.moveaxis(output, 1, -1).reshape(-1, 4)
    output_df = pd.DataFrame(output.tolist(), columns=['gid', 'timestamp', 'prediction', 'label'])
    output_df['gid'] = output_df['gid'].astype(int)

    output_path = os.path.join(kwargs['result_dir'], '{}_{}m_{}_{}_prediction.csv'.format(args.area, args.resolution, args.year, args.months[-1]))
    output_df.to_csv(output_path, index=False) 
    
    
def write_prediction_mongodb(prediction, mapping_mat, args, **kwargs):
    """ write the predictions to a database """

    client = pymongo.MongoClient("mongodb://jon:snow@jonsnow.usc.edu:65533/jonsnow")

    n_times, _, n_rows, n_cols = prediction.shape
    mapping_mat_copy = mapping_mat.reshape(1, n_rows, n_cols)

    # get time list
    min_time = args.min_time
    max_time = args.max_time
    tz = pytz.timezone('America/Los_Angeles')
    time_list = pd.date_range(start=min_time, end=max_time, closed='left', freq='1H')
    time_list = sorted(list(set([tz.localize(x) for x in time_list])))
    for i, t in enumerate(time_list[-n_times:]):
        output = np.concatenate([mapping_mat_copy, prediction[i, ...]], axis=0)
        output = np.moveaxis(output, 0, -1).reshape(-1, 2).tolist()
        output = [{'gid': int(output[i][0]), 'pm25': round(output[i][1], 5)} for i in range(len(output))]
        document = {'timestamp': t, 'data': output}
        utc_time = t.astimezone(pytz.timezone('UTC'))
        write_mongodb(utc_time, document, client, f'la_500m_{utc_time.year}')
    
    client.close()


def write_mongodb(t, document, client, collection_name, max_attempts=3):

    db = client['jonsnow']
    collection = db[collection_name]
    condition = {'timestamp': t}
    
    for i in range(max_attempts):
        try:
            data = collection.find_one(condition)
            if data is not None:
                collection.update_one(condition, {'$set': document})
            else:
                collection.insert_one(document)
            break
            
        except Exception as e:
            print(max_attempts, condition, e)
            time.sleep(5)
            

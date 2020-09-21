import os
import numpy as np
import pandas as pd
import argparse
import torch
import torch.utils.data as dat

from scripts.data_loader import DataObj, load_train_val_test
from scripts.result_viz import spatial_viz, temporal_viz, output_prediction
from utils.metrics import normalize_mat, compute_error


def predict(dap, data_obj, args, **kwargs):
    dap.eval()
    predictions = []

    idx = np.array([i for i in range(args.seq_len, data_obj.label_mat.shape[0])])
    idx_dat = dat.TensorDataset(torch.IntTensor(idx))
    test_idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=1, shuffle=False)

    with torch.no_grad():
        def construct_sequence_x(idx_list, dynamic_x, static_x):
            d_x = [dynamic_x[i - args.seq_len: i + 1, ...] for i in idx_list]
            d_x = np.stack(d_x, axis=0)
            s_x = np.expand_dims(static_x, axis=0)
            s_x = np.repeat(s_x, args.seq_len + 1, axis=1)  # (t, c, h, w)
            s_x = np.repeat(s_x, len(idx_list), axis=0)  # (b, t, c, h, w)
            x = np.concatenate([d_x, s_x], axis=2)
            return torch.FloatTensor(x).to(kwargs['device'])

        for i, data in enumerate(test_idx_data_loader):
            batch_idx = data[0]
            batch_x = construct_sequence_x(batch_idx, data_obj.dynamic_x, data_obj.static_x)  # x = (b, t, c, h, w)
        out, _, _ = dap(batch_x)
        predictions.append(out.cpu().data.numpy())

    prediction = np.concatenate(predictions)
    prediction[prediction <= 0.0] = 0.01

    val_rmse, val_mape, val_r2 = compute_error(data_obj.val_y[args.seq_len:, ...], prediction)
    rmse, mape, r2 = compute_error(data_obj.test_y[args.seq_len:, ...], prediction)

    print(kwargs['model_name'] + f' VAL RMSE = {val_rmse:.6f}, MAPE = {val_mape:.6f}, R2 = {val_r2:.6f}.')
    print(kwargs['model_name'] + f' TEST RMSE = {rmse:.6f}, MAPE = {mape:.6f}, R2 = {r2:.6f}.')
    return prediction


def extract_weights(dap, data_obj, args):
    """ print out the selected features """

    params = [p for p in dap.mask_layer.linear_layer.parameters()]

    weights = []
    weight = params[0].data.cpu().numpy()
    for i in range(weight.shape[0]):
        weights.append(weight[i, i])

    features = data_obj.dynamic_feature_names + data_obj.static_feature_names

    feature_weight = {}
    for i, w in enumerate(weights):
        if abs(w) >= args.mask_thr:
            feature_weight[features[i]] = abs(w)

    sorted_feature_weights = [[k, v] for k, v in sorted(feature_weight.items(), key=lambda item: -item[1])]
    return sorted_feature_weights


def main(args, **kwargs):
    """ load data object """
    data_file = os.path.join(kwargs['data_dir'],
                             '{}_{}m_{}_{}.npz'.format(args.area, args.resolution, args.year, args.date))
    data = np.load(data_file)

    """ extract information from target time period """
    label_mat = data['label_mat']
    dynamic_mat = data['dynamic_mat']
    static_mat = data['static_mat']
    mapping_mat = data['mapping_mat']
    dynamic_features, static_features = list(data['dynamic_features']), list(data['static_features'])

    data_obj = DataObj(label_mat, dynamic_mat, static_mat,
                       None, None, None,
                       dynamic_features, static_features, mapping_mat)

    """ load train, val, test locations """
    data_obj.train_loc, data_obj.val_loc, data_obj.test_loc = load_train_val_test(kwargs['train_val_test_file'], args)

    data_obj.train_y = data_obj.gen_train_val_test_label(data_obj.label_mat, data_obj.train_loc)
    data_obj.val_y = data_obj.gen_train_val_test_label(data_obj.label_mat, data_obj.val_loc)
    data_obj.test_y = data_obj.gen_train_val_test_label(data_obj.tar_label_mat, data_obj.test_loc)

    """ normalize data """
    data_obj.dynamic_x = normalize_mat(data_obj.dynamic_mat, if_retain_last_dim=True)
    data_obj.static_x = normalize_mat(data_obj.static_mat, if_retain_last_dim=True)

    """ load dap model """
    dap = torch.load(kwargs['model_file'])

    """ generate prediction """
    prediction = predict(dap, data_obj, args, **kwargs)
    print('Max: {}; Min: {}'.format(np.max(prediction), np.min(prediction)))

    """ generate weights for the selected features """
    sorted_feature_weights = extract_weights(dap, data_obj, args)
    print(len(sorted_feature_weights), sorted_feature_weights)

    """ draw the spatial prediction """
    if kwargs['if_spatial_viz']:
        mean_prediction = np.mean(prediction, axis=0)
        spatial_viz(mean_prediction, data_obj, args, **kwargs)

    """ draw the temporal predictions """
    if kwargs['if_temporal_viz']:
        loc = data_obj.test_loc[3]
        temporal_viz(prediction, loc, data_obj, args)

    """ Write the output """
    if kwargs['if_output_results']:
        output_prediction(prediction, data_obj, args, **kwargs)


class Param:
    def __init__(self, model_name):
        """ default configuration """
        self.area = 'los_angeles'
        self.resolution = 500
        self.year = 2018
        self.date = model_name.split('__')[3]

        """ hyper parameters """
        self.seq_len = int(model_name.split('__')[4].split('_')[0])
        self.mask_thr = float(
            model_name.split('__')[4].split('_')[2][0] + '.' + model_name.split('__')[4].split('_')[2][1:])


if __name__ == '__main__':

    test_month, min_time, max_time = '201810', '2018-10-01 00:00:00', '2018-11-01 00:00:00'

    base_dir = 'data/'
    model_dir = os.path.join(base_dir, 'models_1/')
    data_dir = '/home/yijun/notebooks/training_data/los_angeles_500m_2018/'
    train_val_test_file = 'data/data/los_angeles_500m_train_val_test.json'
    result_dir = os.path.join(base_dir, 'results/')
    device = torch.device("cuda:3" if torch.cuda.is_available() else 'cpu')  # the gpu device
    kwargs = dict()

    for subdir, dirs, files in os.walk(model_dir):
        for file in files:
            if file.split('__')[3] == test_month and 'dap' in file:
                model_name = file[:-4]
                model_file = os.path.join(subdir, file)
                param = Param(model_name)

                kwargs['base_dir'] = base_dir
                kwargs['data_dir'] = data_dir
                kwargs['result_dir'] = result_dir
                kwargs['model_name'] = model_name
                kwargs['model_file'] = model_file
                kwargs['train_val_test_file']: train_val_test_file
                kwargs['if_spatial_viz'] = True
                kwargs['if_temporal_viz'] = False
                kwargs['if_output_results'] = True
                kwargs['device'] = device
                kwargs['min_time'], kwargs['max_time'] = min_time, max_time
                main(param, **kwargs)

# parser = argparse.ArgumentParser()
#
# """ default configuration """
# parser.add_argument('--area', type=str, default='los_angeles')
# parser.add_argument('--date', type=str, default='201801')
# parser.add_argument('--resolution', type=int, default=500)
#
# parser.add_argument('--seq_len', type=int, default=6)
# parser.add_argument('--mask_thr', type=float, default=0.0001)
#
# args = parser.parse_args()
#
# model_name = 'dap_los_angeles_500_201801_#_6_1_00001_#_1_001_1_#_16_32_6416_13'
# args.seq_len = int(model_name.split('_')[7])
# args.mask_thr = float(model_name.split('_')[9][0] + '.' + model_name.split('_')[9][1:])

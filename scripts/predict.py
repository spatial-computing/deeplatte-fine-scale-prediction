import os
import numpy as np
import pandas as pd
import json
import argparse
import torch
import torch.utils.data as dat

import torch.nn as nn

from models.conv_lstm import ConvLSTM
from models.fc import FC
from models.auto_encoder import AutoEncoder
from models.mask_net import MaskNet

from scripts.data_loader import DataObj
from utils.metrics import normalize_mat, compute_error


def predict(dapm, data_obj, args, **kwargs):
    
    dapm.eval()
    predictions = []

    idx = np.array([i for i in range(args.seq_len, data_obj.test_y.shape[0])])
    idx_dat = dat.TensorDataset(torch.tensor(idx, dtype=torch.int32))
    test_idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=1, shuffle=False)

    with torch.no_grad():
        
        def construct_sequence_x(idx_list, dynamic_x, static_x):
            d_x = [dynamic_x[i - args.seq_len: i + 1, ...] for i in idx_list]
            d_x = np.stack(d_x, axis=0)
            s_x = np.expand_dims(static_x, axis=0)
            s_x = np.repeat(s_x, args.seq_len + 1, axis=1)  # (t, c, h, w)
            s_x = np.repeat(s_x, len(idx_list), axis=0)  # (b, t, c, h, w)
            x = np.concatenate([d_x, s_x], axis=2)
            return torch.tensor(x, dtype=torch.float).to(kwargs['device'])
        
        for i, data in enumerate(test_idx_data_loader):
            batch_idx = data[0]
            batch_x = construct_sequence_x(batch_idx, data_obj.dynamic_x, data_obj.static_x)  # x = (b, t, c, h, w)
            out, _, _, _, _ = dapm(batch_x)
            predictions.append(out.cpu().data.numpy())

    prediction = np.concatenate(predictions)
    prediction[prediction <= 0.0] = 0.01

    val_rmse, val_mape, val_r2 = compute_error(data_obj.val_y[args.seq_len:, ...], prediction)
    rmse, mape, r2 = compute_error(data_obj.test_y[args.seq_len:, ...], prediction)
    
    print(kwargs['model_name'] + f' VAL RMSE = {val_rmse:.4f}, MAPE = {val_mape:.4f}, R2 = {val_r2:.4f}.')
    print(kwargs['model_name'] + f' TEST RMSE = {rmse:.4f}, MAPE = {mape:.4f}, R2 = {r2:.4f}.')
    
    return prediction


def predict_meo_only(dapm, data_obj, args, **kwargs):
    
    dapm.eval()
    predictions = []

    idx = np.array([i for i in range(args.seq_len, data_obj.test_y.shape[0])])
    idx_dat = dat.TensorDataset(torch.IntTensor(idx))
    test_idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=1, shuffle=False)

    with torch.no_grad():
        
        def construct_sequence_x(idx_list, dynamic_x):
            d_x = [dynamic_x[i - args.seq_len: i + 1, ...] for i in idx_list]
            d_x = np.stack(d_x, axis=0)
            return torch.FloatTensor(d_x).to(kwargs['device'])
        
        for i, data in enumerate(test_idx_data_loader):
            batch_idx = data[0]
            batch_x = construct_sequence_x(batch_idx, data_obj.dynamic_x)  # x = (b, t, c, h, w)
            out, _, _ = dapm(batch_x)
            predictions.append(out.cpu().data.numpy())

    prediction = np.concatenate(predictions)
    prediction[prediction <= 0.0] = 0.01

    val_rmse, val_mape, val_r2 = compute_error(data_obj.val_y[args.seq_len:, ...], prediction)
    rmse, mape, r2 = compute_error(data_obj.test_y[args.seq_len:, ...], prediction)
    
    print(kwargs['model_name'] + f' VAL RMSE = {val_rmse:.4f}, MAPE = {val_mape:.4f}, R2 = {val_r2:.4f}.')
    print(kwargs['model_name'] + f' TEST RMSE = {rmse:.4f}, MAPE = {mape:.4f}, R2 = {r2:.4f}.')
    
    return prediction
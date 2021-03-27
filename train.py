import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as dat
from tensorboardX import SummaryWriter
from torch import autograd

import argparse
from prisms_data_preprocessing.gen_train_data import gen_train_data
from options.train_options import TrainOptions
from utils.early_stopping import EarlyStopping
from utils.metrics import compute_error
from models.spatial_loss_func import SpatialLossFunc

from scipy.optimize import curve_fit


def gaussian(h, r, s, n=0):
    return n + s * (1. - np.exp(- (h ** 2 / (r / 2.) ** 2)))


def get_fit_bounds(x, y):
    n = np.nanmin(y)
    r = np.nanmax(x)
    s = np.nanmax(y)
    return (0, [r, s, n])


def get_fit_func(x, y, model):
    try:
        bounds = get_fit_bounds(x, y)
        popt, _ = curve_fit(model, x, y, method='trf', p0=bounds[1], bounds=bounds)
        return popt
    except Exception as e:
        return [0, 0, 0]


def gen_semivariogram(distances, variances, bins, thr):
    valid_variances, valid_bins = [], []
    for b in range(len(bins) - 1):
        left, right = bins[b], bins[b + 1]
        mask = (distances >= left) & (distances < right)
        if np.count_nonzero(mask) > thr:
            v = np.nanmean(variances[mask])
            d = np.nanmean(distances[mask])
            valid_variances.append(v)
            valid_bins.append(d)

    x, y = np.array(valid_bins), np.array(valid_variances)
    popt = get_fit_func(x, y, model=gaussian)
    return popt


def train(dapm, data_obj, args, **kwargs):
    """ construct index-based data loader """

    idx = np.array([i for i in range(args.seq_len + 1, data_obj.train_y.shape[0])])
    idx_dat = dat.TensorDataset(torch.tensor(idx, dtype=torch.int32))
    train_idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=args.batch_size, shuffle=True)

    idx = np.array([i for i in range(args.seq_len, data_obj.test_y.shape[0])])
    idx_dat = dat.TensorDataset(torch.tensor(idx, dtype=torch.int32))
    test_idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=1, shuffle=False)

    """ set writer, loss function, and optimizer """
    writer = SummaryWriter(kwargs['run_file'])
    loss_func = nn.MSELoss()
    spatial_loss_func = SpatialLossFunc(sp_neighbor=args.sp_neighbor)
    optimizer = optim.Adam(dapm.parameters(), lr=args.lr, weight_decay=1e-8)
    early_stopping = EarlyStopping(patience=7, verbose=True)

    for epoch in range(args.epochs):

        dapm.train()
        total_losses, train_losses, val_losses, alosses = [], [], [], []

        for _, idx in enumerate(train_idx_data_loader):
            batch_idx = idx[0]

            ############################
            # construct sequence input #
            ############################

            def construct_sequence_x(idx_list, dynamic_x, static_x):
                d_x = [dynamic_x[i - args.seq_len: i + 1, ...] for i in idx_list]
                d_x = np.stack(d_x, axis=0)
                s_x = np.expand_dims(static_x, axis=0)
                s_x = np.repeat(s_x, args.seq_len + 1, axis=1)  # (t, c, h, w)
                s_x = np.repeat(s_x, len(idx_list), axis=0)  # (b, t, c, h, w)
                x = np.concatenate([d_x, s_x], axis=2)
                return torch.tensor(x, dtype=torch.float).to(kwargs['device'])

            def construct_y(idx_list, output_y):
                y = [output_y[i] for i in idx_list]
                y = np.stack(y, axis=0)
                return torch.tensor(y, dtype=torch.float).to(kwargs['device'])

            batch_x = construct_sequence_x(batch_idx, data_obj.dynamic_x, data_obj.static_x)  # x = (b, t, c, h, w)
            batch_y = construct_y(batch_idx, data_obj.train_y)  # y = (b, c, h, w)
            batch_val_y = construct_y(batch_idx, data_obj.val_y)

            ###################
            # train the model #
            ###################

            out, masked_x, _, de_x, em = dapm(batch_x)
            train_loss = loss_func(batch_y[~torch.isnan(batch_y)], out[~torch.isnan(batch_y)])
            train_losses.append(train_loss.item())

            # add loss according to the model type
            total_loss = train_loss
            if 'sp' in args.model_type:
                mask_layer_params = torch.cat([x.view(-1) for x in dapm.mask_layer.parameters()])
                l1_regularization = torch.norm(mask_layer_params, 1)
                total_loss += l1_regularization * args.alpha

            if 'ae' in args.model_type:
                ae_loss = loss_func(masked_x, de_x)
                total_loss += ae_loss * args.gamma

            if 'sc' in args.model_type:
                sp_loss = spatial_loss_func(out)
                total_loss += sp_loss * args.beta

            if 'embedding_spatiotemporal_similar' in args.model_type:
                # 1-step temporal neighboring loss
                pre_batch_idx = batch_idx - torch.ones_like(batch_idx)
                pre_batch_x = construct_sequence_x(pre_batch_idx, data_obj.dynamic_x,
                                                   data_obj.static_x)  # x = (b, t, c, h, w)
                _, _, _, _, pre_em = dapm(pre_batch_x)
                tp_loss = torch.mean(torch.mean((em - pre_em) ** 2, axis=1))

                # 1-step spatial neighboring loss
                sp_loss = 0.
                sp_loss += torch.mean(torch.mean((em[..., 1:, 1:] - em[..., :-1, :-1]) ** 2, axis=1))
                sp_loss += torch.mean(torch.mean((em[..., 1:, :] - em[..., :-1, :]) ** 2, axis=1))
                sp_loss += torch.mean(torch.mean((em[..., :, 1:] - em[..., :, :-1]) ** 2, axis=1))
                alosses.append(tp_loss.item() + sp_loss.item())
                total_loss += (tp_loss + sp_loss) * args.eta

            total_losses.append(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            ######################
            # validate the model #
            ######################

            val_loss = loss_func(batch_val_y[~torch.isnan(batch_val_y)], out[~torch.isnan(batch_val_y)])
            val_losses.append(val_loss.item())

        avg_total_loss = np.average(total_losses)
        avg_train_loss = np.average(train_losses)
        avg_a_loss = np.average(alosses)
        avg_val_loss = np.average(val_losses)

        # write for tensorboard visualization
        writer.add_scalar('data/train_loss', avg_total_loss, epoch)
        writer.add_scalar('data/val_loss', avg_val_loss, epoch)

        logging.info(
            f'Epoch [{epoch}/{args.epochs}] total_loss = {avg_total_loss:.4f}, train_loss = {avg_train_loss:.4f}, a_loss = {avg_a_loss:.4f}, valid_loss = {avg_val_loss:.4f}.')

        ##################
        # early_stopping #
        ##################

        early_stopping(avg_val_loss, dapm, kwargs['model_file'])

        #########################
        # evaluate testing data #
        #########################

        if early_stopping.counter < 2 and epoch % 2 == 0:

            dapm.eval()
            predictions = []

            with torch.no_grad():
                for i, data in enumerate(test_idx_data_loader):
                    batch_idx = data[0]
                    batch_x = construct_sequence_x(batch_idx, data_obj.dynamic_x,
                                                   data_obj.static_x)  # x = (b, t, c, h, w)
                    out, _, _, _, _ = dapm(batch_x)
                    predictions.append(out.cpu().data.numpy())

            prediction = np.concatenate(predictions)
            rmse, mape, r2 = compute_error(data_obj.test_y[args.seq_len:, ...], prediction)
            writer.add_scalar('data/test_rmse', rmse, epoch)
            logging.info(f'Testing: RMSE = {rmse:.4f}, MAPE = {mape:.4f}, R2 = {r2:.4f}.')

        if early_stopping.early_stop:
            logging.info(kwargs['model_name'] + f' val_loss = {early_stopping.val_loss_min:.4f}.')
            logging.info('Early stopping')
            break


if __name__ == '__main__':
    trainoptions = TrainOptions()
    trainoptions.initialize()

    parser = trainoptions.parser
    args = trainoptions.parse(parser)
    # args = TrainOptions().parse()

    print(args['max_time'],args['min_time'])
    data = gen_train_data(args['min_time'],args['max_time'])
    train(data)




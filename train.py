import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dat
from torch.utils.tensorboard import SummaryWriter

from scripts.data_loader import load_data_from_file, load_data_from_db
from options.train_options import parse_args, verbose
from utils.early_stopping import EarlyStopping
from utils.metrics import compute_error
from models.deeplatte import DeepLatte
from models.loss import SpatialLoss, TemporalLoss


def train():
    """ train """

    """ construct index-based data loader """
    idx = np.array([i for i in range(args.seq_len + 1, data_obj.num_times)])
    idx_dat = dat.TensorDataset(torch.tensor(idx, dtype=torch.int32))
    train_idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=args.batch_size, shuffle=True)

    idx = np.array([i for i in range(args.seq_len + 1, data_obj.num_times)])
    idx_dat = dat.TensorDataset(torch.tensor(idx, dtype=torch.int32))
    test_idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=1, shuffle=False)

    """ set writer, loss function, and optimizer """
    writer = SummaryWriter(args.run_file)
    mse_loss_func = nn.MSELoss()
    spatial_loss_func = SpatialLoss(sp_neighbor=args.sp_neighbor)
    temporal_loss_func = TemporalLoss(tp_neighbor=args.tp_neighbor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience, verbose=args.verbose)

    def construct_sequence_x(idx_list, dynamic_x, static_x):
        d_x = [dynamic_x[i - args.seq_len + 1: i + 1, ...] for i in idx_list]
        d_x = np.stack(d_x, axis=0)
        s_x = np.expand_dims(static_x, axis=0)
        s_x = np.repeat(s_x, args.seq_len, axis=1)  # shape: (t, c, h, w)
        s_x = np.repeat(s_x, len(idx_list), axis=0)  # shape: (b, t, c, h, w)
        x = np.concatenate([d_x, s_x], axis=2)
        return torch.tensor(x, dtype=torch.float).to(device)

    def construct_y(idx_list, output_y):
        y = [output_y[i] for i in idx_list]
        y = np.stack(y, axis=0)
        return torch.tensor(y, dtype=torch.float).to(device)

    """ training """
    for epoch in range(args.epochs):

        model.train()
        total_losses, train_losses, val_losses, alosses = 0, 0, 0, 0

        for _, idx in enumerate(train_idx_data_loader):
            batch_idx = idx[0]

            """ construct sequence input """
            batch_x = construct_sequence_x(batch_idx, data_obj.dynamic_x, data_obj.static_x)  # shape: (b, t, c, h, w)
            batch_y = construct_y(batch_idx, data_obj.train_y)  # shape: (b, 1, h, w)
            batch_val_y = construct_y(batch_idx, data_obj.val_y)

            """ start train """
            out, masked_x, _, de_x, em = model(batch_x)
            train_loss = mse_loss_func(batch_y[~torch.isnan(batch_y)], out[~torch.isnan(batch_y)])
            train_losses += train_loss.item()

            """ add loss according to the model type """
            total_loss = train_loss
            if 'l1' in args.model_type:
                total_loss += model.sparse_layer.l1_loss() * args.alpha

            if 'ae' in args.model_type:
                ae_loss = mse_loss_func(masked_x, de_x)
                total_loss += ae_loss * args.gamma

            if 'sc' in args.model_type:
                sp_loss = spatial_loss_func(out)
                total_loss += sp_loss * args.beta

            # if 'vg' in args.model_type:
            #     # 1-step temporal neighboring loss
            #     pre_batch_idx = batch_idx - torch.ones_like(batch_idx)
            #     pre_batch_x = construct_sequence_x(pre_batch_idx, data_obj.dynamic_x,
            #                                        data_obj.static_x)  # x = (b, t, c, h, w)
            #     _, _, _, _, pre_em = model(pre_batch_x)
            #     tp_loss = torch.mean(torch.mean((em - pre_em) ** 2, axis=1))
            #
            #     # 1-step spatial neighboring loss
            #     sp_loss = 0.
            #     sp_loss += torch.mean(torch.mean((em[..., 1:, 1:] - em[..., :-1, :-1]) ** 2, axis=1))
            #     sp_loss += torch.mean(torch.mean((em[..., 1:, :] - em[..., :-1, :]) ** 2, axis=1))
            #     sp_loss += torch.mean(torch.mean((em[..., :, 1:] - em[..., :, :-1]) ** 2, axis=1))
            #     alosses.append(tp_loss.item() + sp_loss.item())
            #     total_loss += (tp_loss + sp_loss) * args.eta

            total_losses += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            """ validate """
            val_loss = mse_loss_func(batch_val_y[~torch.isnan(batch_val_y)], out[~torch.isnan(batch_val_y)])
            val_losses += val_loss.item()

            break

        # write for tensor board visualization
        writer.add_scalar('data/train_loss', train_losses, epoch)
        writer.add_scalar('data/val_loss', val_losses, epoch)

        # early_stopping
        early_stopping(val_losses, model, model_file)

        # evaluate testing data
        if epoch % 2 == 0 and len(data_obj.test_loc) == 0:

            model.eval()
            prediction = []

            with torch.no_grad():
                for i, data in enumerate(test_idx_data_loader):
                    batch_idx = data[0]
                    batch_x = construct_sequence_x(batch_idx, data_obj.dynamic_x, data_obj.static_x)  # (b, t, c, h, w)
                    out, _, _, _, _ = model(batch_x)
                    prediction.append(out.cpu().data.numpy())

            prediction = np.concatenate(prediction)
            acc = compute_error(data_obj.test_y[args.seq_len + 1:, ...], prediction)

        if args.verbose:
            logging.info('Epoch [{}/{}] testing: rmse = {:.3f}, mape = {:.3f}, r2 = {:.3f}.'
                         .format(epoch, args.epochs, *acc))
            logging.info('Epoch [{}/{}] total_loss = {:.3f}, train_loss = {:.3f}, val_loss = {:.3f}.'
                         .format(epoch, args.epochs, total_losses, train_losses, val_losses))

        if early_stopping.early_stop:
            logging.info('{} - val_loss = {:.4f}. Early stopping.'.format(args.model_name, early_stopping.val_loss_min))
            break


if __name__ == '__main__':

    args = parse_args()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')  # the gpu device

    """ load data """
    if os.path.exists(args.data_path):
        data_obj = load_data_from_file(args.data_path)
    else:
        data_obj = load_data_from_db(args)

    """ load model """
    model = DeepLatte(in_features=data_obj.num_features,
                      en_features=[64, 16],
                      de_features=[16, 64],
                      in_size=(data_obj.num_rows, data_obj.num_cols),
                      h_channels=32,
                      kernel_sizes=[1, 3, 5],
                      num_layers=1,
                      h_features=16,
                      out_features=1,  # fixed
                      p=0.5,
                      device=device).to(device)

    model_file = os.path.join(args.model_path, args.model_name + '.pkl')
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))

    train()




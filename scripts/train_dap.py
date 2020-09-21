import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dat
from tensorboardX import SummaryWriter

from utils.early_stopping import EarlyStopping
from utils.metrics import compute_error


class SpatialLoss(nn.Module):

    def __init__(self, sp_neighbor):
        super(SpatialLoss, self).__init__()
        self.sp_neighbor = sp_neighbor

    def forward(self, x):

        loss = 0.0
        n_times, _, n_rows, n_cols = x.shape

        for i in range(-self.sp_neighbor, self.sp_neighbor + 1):
            for j in range(-self.sp_neighbor, self.sp_neighbor + 1):
                weight = (i * i + j * j) ** 0.5
                if i >= 0 and j >= 0 and weight != 0:
                    loss += torch.sum(torch.pow((x[..., i:, j:] - x[..., : n_rows - i, : n_cols - j]), 2)) / weight
                elif i >= 0 and j < 0:
                    loss += torch.sum(torch.pow((x[..., i:, :j] - x[..., : n_rows - i, -j:]), 2)) / weight
                elif i < 0 and j >= 0:
                    loss += torch.sum(torch.pow((x[..., :i, j:] - x[..., -i:, : n_cols - j]), 2)) / weight
                elif i < 0 and j < 0:
                    loss += torch.sum(torch.pow((x[..., :i, :j] - x[..., -i:, -j:]), 2)) / weight
                else:
                    pass

        return loss / n_times / n_rows / n_cols


def train(dap, data_obj, args, **kwargs):
    """ construct index-based data loader """
    idx = np.array([i for i in range(args.seq_len, data_obj.n_times)])
    idx_dat = dat.TensorDataset(torch.IntTensor(idx))
    train_idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=args.batch_size, shuffle=True)

    idx = np.array([i for i in range(args.seq_len, data_obj.test_y.shape[0])])
    idx_dat = dat.TensorDataset(torch.IntTensor(idx))
    test_idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=1, shuffle=False)

    """ set writer, loss function, and optimizer """
    writer = SummaryWriter(kwargs['run_file'])
    loss_func = nn.MSELoss()
    spatial_loss_func = SpatialLoss(sp_neighbor=args.sp_neighbor)
    optimizer = optim.Adam(dap.parameters(), lr=args.lr, weight_decay=1e-8)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(args.epochs):

        dap.train()
        total_losses, train_losses, val_losses = [], [], []

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
                return torch.FloatTensor(x).to(kwargs['device'])

            def construct_y(idx_list, output_y):
                y = [output_y[i] for i in idx_list]
                y = np.stack(y, axis=0)
                return torch.FloatTensor(y).to(kwargs['device'])

            batch_x = construct_sequence_x(batch_idx, data_obj.dynamic_x, data_obj.static_x)  # x = (b, t, c, h, w)
            batch_y = construct_y(batch_idx, data_obj.train_y)  # y = (b, c, h, w)
            batch_val_y = construct_y(batch_idx, data_obj.val_y)

            ###################
            # train the model #
            ###################

            out, masked_x, de_x = dap(batch_x)
            sp_loss = spatial_loss_func(out)

            mask_layer_params = torch.cat([x.view(-1) for x in dap.mask_layer.parameters()])
            l1_regularization = torch.norm(mask_layer_params, 1)
            ae_loss = loss_func(masked_x, de_x)

            loss = loss_func(batch_y[~torch.isnan(batch_y)], out[~torch.isnan(batch_y)])
            train_losses.append(loss.item())

            total_loss = loss + l1_regularization * args.alpha + sp_loss * args.beta + ae_loss * args.gamma
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
        avg_val_loss = np.average(val_losses)

        # write for tensorboard visualization
        writer.add_scalar('data/{}_train_loss'.format(kwargs['model_name']), avg_total_loss, epoch)
        writer.add_scalar('data/{}_val_loss'.format(kwargs['model_name']), avg_val_loss, epoch)

        logging.info(f'Epoch [{epoch}/{args.epochs}] total_loss = {avg_total_loss:.6f}, '
                     f'train_loss = {avg_train_loss:.6f}, valid_loss = {avg_val_loss:.6f}.')

        ##################
        # early_stopping #
        ##################

        early_stopping(avg_val_loss, dap, kwargs['model_file'])

        #########################
        # evaluate testing data #
        #########################

        if early_stopping.counter < 2 and epoch % 2 == 0:

            dap.eval()
            predictions = []

            with torch.no_grad():
                for i, data in enumerate(test_idx_data_loader):
                    batch_idx = data[0]
                    batch_x = construct_sequence_x(batch_idx, data_obj.tar_dynamic_x,
                                                   data_obj.tar_static_x)  # x = (b, t, c, h, w)
                    out, _, _ = dap(batch_x)
                    predictions.append(out.cpu().data.numpy())

            predictions = np.concatenate(predictions)
            rmse, mape, r2 = compute_error(data_obj.test_y[args.seq_len:, ...], predictions)
            logging.info(f'Testing: RMSE = {rmse:.6f}, MAPE = {mape:.6f}, R2 = {r2:.6f}.')

        if early_stopping.early_stop:
            logging.info(kwargs['model_name'] + f' val_loss = {early_stopping.val_loss_min:.6f}.')
            logging.info('Early stopping')
            break

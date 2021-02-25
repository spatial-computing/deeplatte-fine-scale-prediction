import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dat
from tensorboardX import SummaryWriter

from utils.early_stopping import EarlyStopping
from utils.metrics import compute_error
from models.spatial_loss_func import SpatialLossFunc


def train(dapm, data_obj, args, **kwargs):
    
    """ construct index-based data loader """
    idx = np.array([i for i in range(args.seq_len, data_obj.train_y.shape[0])])
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
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1, last_epoch=4000)
    early_stopping = EarlyStopping(patience=7, verbose=True)

    for epoch in range(args.epochs):

        dapm.train()
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

            out, masked_x, _, de_x, _ = dapm(batch_x)
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

            total_losses.append(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
#             scheduler.step()

            ######################
            # validate the model #
            ######################

            val_loss = loss_func(batch_val_y[~torch.isnan(batch_val_y)], out[~torch.isnan(batch_val_y)])
            val_losses.append(val_loss.item())
        

        avg_total_loss = np.average(total_losses)
        avg_train_loss = np.average(train_losses)
        avg_val_loss = np.average(val_losses)

        # write for tensorboard visualization
        writer.add_scalar('data/train_loss', avg_total_loss, epoch)
        writer.add_scalar('data/val_loss', avg_val_loss, epoch)

        logging.info(f'Epoch [{epoch}/{args.epochs}] total_loss = {avg_total_loss:.4f}, train_loss = {avg_train_loss:.4f}, valid_loss = {avg_val_loss:.4f}.')

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
                    batch_x = construct_sequence_x(batch_idx, data_obj.dynamic_x, data_obj.static_x)  # x = (b, t, c, h, w)
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

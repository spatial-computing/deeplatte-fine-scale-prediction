import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dat
from tensorboardX import SummaryWriter


def train(ae, data_obj, args, **kwargs):

    print('>>> {}: Training process has started.'.format(kwargs['model_name']))

    """ construct index-based data loader """
    idx = np.array([i for i in range(data_obj.n_times)])
    idx_dat = dat.TensorDataset(torch.tensor(idx, dtype=torch.int32))
    idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=args.batch_size, shuffle=True)

    """ set writer, loss function, and optimizer """
    writer = SummaryWriter(kwargs['run_file'])
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=args.lr, weight_decay=1e-8)
    
    for epoch in range(args.epochs):

        train_losses = []
        for _, idx in enumerate(idx_data_loader):

            batch_idx = idx[0]

            batch_dynamic_x = data_obj.dynamic_x[batch_idx]
            batch_static_x = np.repeat(data_obj.static_x, len(batch_idx), axis=0)  # (batch_size, n_rows, n_cols, n_features)
            batch_x = np.concatenate([batch_dynamic_x, batch_static_x], axis=-1)
            batch_x = torch.tensor(batch_x, dtype=torch.float).to(kwargs['device'])  # x = (b, t, c, h, w)

            _, de_x = ae(batch_x)
            loss = loss_func(batch_x, de_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_loss = sum(train_losses) / len(train_losses)
        writer.add_scalar('data/{}_loss'.format(kwargs['model_name']), avg_loss, epoch)

        if epoch % 5 == 0:
            print('Epoch [{}/{}], Loss = {}.'.format(epoch, args.epochs, avg_loss))

    torch.save(ae, kwargs['model_file'])
    print('>>> {} has been saved.'.format(kwargs['model_name']))

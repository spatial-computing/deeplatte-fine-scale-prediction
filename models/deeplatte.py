import torch
import torch.nn as nn

from models.convlstm import ConvLSTM
from models.autoencoder import AutoEncoder
from models.linear import DiagPruneLinear, Stack2Linear


class DeepLatte(nn.Module):

    def __init__(self, in_features, en_features, de_features,
                 in_size, h_channels, kernel_sizes, num_layers,
                 h_features, out_features, **kwargs):
        """
        params:
            in_features (int): size of input sample
            en_features (list): list of number of features for the encoder layers
            de_features (list): list of number of features for the decoder layers
            in_size (int, int): height and width of input tensor as (height, width)
            h_channels (int or list): number of channels of hidden state, assert len(h_channels) == num_layers
            kernel_sizes (list): size of the convolution kernels
            num_layers (int): number of layers in ConvLSTM
            h_features (int): size of hidden features in the FC layer
            out_features (int): size of output sample
        """

        super(DeepLatte, self).__init__()

        self.kwargs = kwargs
        self.device = kwargs.get('device', 'cpu')

        # sparse layer
        self.sparse_layer = DiagPruneLinear(in_features=in_features, device=self.device)

        # auto_encoder layer
        self.ae = AutoEncoder(in_features=in_features, en_features=en_features, de_features=de_features)

        if kwargs.get('ae_pretrain_weight') is not None:
            self.ae.load_state_dict(kwargs['ae_pretrain_weight'])
            for param in self.ae.parameters():
                param.requires_grad = True

        # conv_lstm layers
        h_channels = self._extend_for_multilayer(h_channels, num_layers)  # len(h_channels) == num_layers
        self.conv_lstm_list = nn.ModuleList()
        for i in kernel_sizes:
            self.conv_lstm_list.append(ConvLSTM(in_size=in_size,
                                                in_channels=en_features[-1],
                                                h_channels=h_channels,
                                                kernel_size=(i, i),
                                                num_layers=num_layers,
                                                batch_first=kwargs.get('batch_first', True),
                                                output_last=kwargs.get('only_last_state', True),
                                                device=self.device))

        self.fc = Stack2Linear(in_features=h_channels[-1] * len(kernel_sizes),
                               h_features=h_features,
                               out_features=out_features)

    def forward(self, input_data):  # shape: (b, t, c, h, w)
        """
        param:
            input_data (batch_size, seq_len, num_channels, height, width)
        """

        batch_size, seq_len, num_channels, _, _ = input_data.shape
        x = input_data.permute(0, 1, 3, 4, 2)  # shape: (b, t, h, w, c), moving feature dimension to the last

        # sparse layer
        sparse_x = self.sparse_layer(x)

        # auto-encoder layer
        en_x, de_x = self.ae(sparse_x)
        en_x = en_x.permute(0, 1, 4, 2, 3)  # shape: (b, t, c, h, w), moving height and weight to the last

        # conv_lstm layers
        conv_lstm_out_list = []
        for conv_lstm in self.conv_lstm_list:
            _, (_, cell_last_state) = conv_lstm(en_x)
            conv_lstm_out_list.append(cell_last_state)

        conv_lstm_out = torch.cat(conv_lstm_out_list, dim=1)  # shape: (b, c, h, w)

        # fully-connected layer
        out = conv_lstm_out.permute(0, 2, 3, 1)  # shape: (b, h, w, c), moving feature dimension to the last
        out = self.fc(out)
        out = out.permute(0, 3, 1, 2)  # shape: (b, c, h, w), moving height and weight to the last

        return out, sparse_x, en_x, de_x, conv_lstm_out

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

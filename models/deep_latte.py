import torch
import torch.nn as nn

from models.conv_lstm import ConvLSTM
from models.fc import FC
from models.auto_encoder import AutoEncoder
from models.mask_net import MaskNet
from models.feature_attention import FeatureAttention


class DeepLatte(nn.Module):

    def __init__(self, in_dim, ae_en_h_dims, ae_de_h_dims,
                 conv_lstm_in_size, conv_lstm_in_dim, conv_lstm_h_dim, conv_lstm_kernel_sizes, conv_lstm_n_layers,
                 fc_in_dim, fc_h_dims, fc_out_dim, **kwargs):

        super(DeepLatte, self).__init__()

        self.kwargs = kwargs
        self.device = kwargs.get('device', 'cpu')

        # sparse layer
        mask_indices = [[range(in_dim)], [range(in_dim)]]
        self.mask_layer = MaskNet(in_dim, in_dim, mask_indices=mask_indices, device=self.device)
        self.mask_thre = kwargs.get('mask_thre', 0.0001)
        
        # attention layer
        self.attention = FeatureAttention(in_dim, in_dim, device=self.device)
        
        # auto_encoder layer
        self.ae = AutoEncoder(in_dim=in_dim,
                              en_h_dims=ae_en_h_dims,
                              de_h_dims=ae_de_h_dims)

        if kwargs.get('ae_pretrain_weight') is not None:
            self.ae.load_state_dict(kwargs['ae_pretrain_weight'])
        else:
            raise ValueError('AutoEncoder not pretrained.')

        for p in self.ae.parameters():
            p.requires_grad = True

        # ConvLSTM layers
        self.conv_lstm_list = nn.ModuleList()
        for i in conv_lstm_kernel_sizes:
            i_kernel_size = (i, i)
            conv_lstm = ConvLSTM(in_size=conv_lstm_in_size,
                                 in_dim=conv_lstm_in_dim,
                                 h_dim=conv_lstm_h_dim,
                                 kernel_size=i_kernel_size,
                                 num_layers=conv_lstm_n_layers,
                                 batch_first=kwargs.get('conv_lstm_batch_first', True),
                                 bias=kwargs.get('conv_lstm_bias', True),
                                 only_last_state=kwargs.get('only_last_state', True),
                                 device=self.device)
            self.conv_lstm_list.append(conv_lstm)

        #########################
        # fully-connected layer #
        #########################

        self.fc = FC(in_dim=fc_in_dim,  # assert in_dim == n_conv_lstm * conv_lstm_h_dim
                     h_dims=fc_h_dims,
                     out_dim=fc_out_dim,
                     p_dropout=kwargs.get('fc_p_dropout', 0.1))

    def forward(self, input_data):  # input_data: (b, t, c, h, w)

        x = input_data.permute(0, 1, 3, 4, 2)  # => (b, t, h, w, c)

        ################
        # sparse layer #
        ################

        masked_x = self.mask_layer(x)
        for p in self.mask_layer.parameters():
            for i in range(p.shape[0]):
                if -self.mask_thre <= p[i, i] <= self.mask_thre:
                    masked_x[..., i] = 0.0

#         masked_x = self.attention(x)

        ######################
        # auto-encoder layer #
        ######################

        en_x, de_x = self.ae(masked_x)        
        en_x = en_x.permute(0, 1, 4, 2, 3)  # => (b, t, c, h, w)

        ####################
        # conv_lstm layers #
        ####################

        for i in range(self.conv_n_layers):

            hidden_states = []
            for t in range(seq_len):
                data = seq_data[t]

                if i == 0:
                    h, c = self.cell_list[i](data.x, data.edge_index, data.edge_weight, H=h, C=c)
                else:
                    h, c = self.cell_list[i](x[:, t, :], data.edge_index, data.edge_weight, H=h, C=c)

                hidden_states.append(h)

            hidden_states = torch.stack(tuple(hidden_states), dim=1)  # hidden_states: [n_nodes x seq_len x h_dim]
            x = hidden_states

        # conv_lstm_out = torch.cat(conv_lstm_out_list, dim=1)  # => (b, c, h, w)

        #########################
        # fully-connected layer #
        #########################

        fc_out = conv_lstm_out.permute(0, 2, 3, 1)  # => (b, h, w, c)
        fc_out = self.fc(fc_out)
        fc_out = fc_out.permute(0, 3, 1, 2)  # => (b, c, h, w)

        return fc_out, masked_x, en_x, de_x, conv_lstm_out

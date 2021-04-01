import torch
import torch.nn as nn

"""
Reference: https://github.com/spacejake/convLSTM.pytorch/blob/master/convlstm.py

"""


class ConvLSTMCell(nn.Module):

    def __init__(self, in_size, in_channels, h_channels, kernel_size, bias=True):
        """
        params:
            in_size (int, int) - height and width of input tensor as (height, width)
            in_channels (int) - number of channels in the input image
            h_channels (int) - number of channels of hidden state
            kernel_size (int, int) - size of the convolution kernel
            bias (bool, optional) - default: True
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = in_size
        self.h_channels = h_channels
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=in_channels + h_channels,
                              out_channels=4 * h_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, input_data, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((input_data, h_prev), dim=1)  # concatenate along channel axis

        combined_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_output, self.h_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)

        return h_cur, c_cur

    def init_hidden(self, batch_size, device):
        """ initialize the first hidden state as zeros """
        return (torch.zeros(batch_size, self.h_channels, self.height, self.width).to(device),
                torch.zeros(batch_size, self.h_channels, self.height, self.width).to(device))


class ConvLSTM(nn.Module):

    def __init__(self, in_size, in_channels, h_channels, kernel_size, num_layers, **kwargs):

        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        self.height, self.width = in_size
        self.num_layers = num_layers
        self.batch_first = kwargs.get('batch_first', True)
        self.output_last = kwargs.get('output_last', True)
        self.device = kwargs.get('device', 'cpu')

        self.cell_list = nn.ModuleList()
        for i in range(0, self.num_layers):
            cur_in_channels = in_channels if i == 0 else h_channels[i - 1]
            self.cell_list.append(ConvLSTMCell(in_size=(self.height, self.width),
                                               in_channels=cur_in_channels,
                                               h_channels=h_channels[i],
                                               kernel_size=kernel_size))

    def forward(self, input_data, hidden_state=None):
        """
        params:
            input_data (batch_size, seq_len, num_channels, height, width)
            hidden_state: None

        return:
            hidden_states_list[-1], last_state_list[-1] | hidden_states_list, last_state_list
        """

        if not self.batch_first:  # (t, b, c, h, w) -> (b, t, c, h, w)
            input_data = input_data.permute(1, 0, 2, 3, 4)

        if hidden_state is None:
            hidden_state = self.get_init_states(batch_size=input_data.size(0))

        hidden_states_list, last_state_list = [], []

        seq_len = input_data.size(1)
        cur_layer_input = input_data

        for i in range(self.num_layers):
            h, c = hidden_state[i]
            hidden_states = []
            for t in range(seq_len):
                h, c = self.cell_list[i](input_data=cur_layer_input[:, t, :, :, :], prev_state=[h, c])
                hidden_states.append(h)

            hidden_states = torch.stack(tuple(hidden_states), dim=1)
            cur_layer_input = hidden_states  # the output of (n) hidden layer is the input of (n+1) hidden layer

            hidden_states_list.append(hidden_states)
            last_state_list.append((h, c))

        if self.output_last:
            return hidden_states_list[-1], last_state_list[-1]
        else:
            return hidden_states_list, last_state_list

    def get_init_states(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, self.device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """ kernel size should be in the tuple mode (k, k) """
        if not isinstance(kernel_size, tuple):
            raise ValueError('kernel_size must be tuple')

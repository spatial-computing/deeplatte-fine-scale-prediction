import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
https://github.com/spacejake/convLSTM.pytorch/blob/master/convlstm.py

"""


class ConvLSTMCell(nn.Module):

    def __init__(self, in_size, in_dim, h_dim, kernel_size, bias):
        """
        Params:
            in_size: (int, int)
                Height and width of input tensor as (height, width)
            in_dim: int
                Number of channels of input tensor
            h_dim: int
                Number of channels of hidden state
            kernel_size: (int, int)
                Size of the convolutional kernel
            bias: bool
                Whether or not to add the bias
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = in_size
        self.input_dim = in_dim
        self.hidden_dim = h_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_data, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((input_data, h_prev), dim=1)  # concatenate along channel axis

        combined_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_output, self.hidden_dim, dim=1)

        i = F.sigmoid(cc_i)
        f = F.sigmoid(cc_f)
        o = F.sigmoid(cc_o)
        g = F.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * F.tanh(c_cur)

        return h_cur, c_cur

    def init_hidden(self, batch_size, device):
        """ initialize the first hidden state as zeros """
        state = (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                 Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))
        state = (state[0].to(device), state[1].to(device))
        return state


class ConvLSTM(nn.Module):

    def __init__(self, in_size, in_dim, h_dim, kernel_size, num_layers, **kwargs):

        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(h_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = in_size
        self.input_dim = in_dim
        self.hidden_dim = h_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = kwargs.get('batch_first', False)
        self.bias = kwargs.get('bias', True)
        self.output_option = kwargs.get('only_last_state', True)
        self.device = kwargs.get('device', 'cpu')

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(in_size=(self.height, self.width),
                                          in_dim=cur_input_dim,
                                          h_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_data, hidden_state=None):
        """
        Params:
            input_tensor: (batch_size, seq_len, n_channel, height, width)
                5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
            hidden_state: None

        Returns:
            last_state_list, layer_output
        """

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_data = input_data.permute(1, 0, 2, 3, 4)

        if hidden_state is None:
            hidden_state = self.get_init_states(batch_size=input_data.size(0))

        hidden_states_list = []
        last_state_list = []

        seq_len = input_data.size(1)
        cur_layer_input = input_data

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            hidden_states = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_data=cur_layer_input[:, t, :, :, :],
                                                 prev_state=[h, c])
                hidden_states.append(h)

            hidden_states = torch.stack(tuple(hidden_states), dim=1)
            cur_layer_input = hidden_states  # the output of (n) hidden layer is the input of (n+1) hidden layer

            hidden_states_list.append(hidden_states)
            last_state_list.append((h, c))

        if self.output_option:
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
            raise ValueError('`kernel_size` must be tuple')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

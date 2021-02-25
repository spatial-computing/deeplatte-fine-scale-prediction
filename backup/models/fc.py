from torch import nn
import torch.nn.functional as F


# a simple Regression model

class FC(nn.Module):

    def __init__(self, in_dim, h_dims, out_dim, **kwargs):

        super(FC, self).__init__()

        # define parameters

        self.input_dim = in_dim
        self.hidden_dims = h_dims
        self.output_dim = out_dim
        self.p_dropout = kwargs.get('p_dropout', 0.0)

        # input layer
        self.in_layer = nn.Linear(self.input_dim, self.hidden_dims[0])

        # hidden layers
        self.hidden_layers = nn.ModuleList()
        for k in range(len(self.hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(self.hidden_dims[k], self.hidden_dims[k + 1]))

        # output layer
        self.dropout = nn.Dropout(p=self.p_dropout)
        self.out_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)

    def forward(self, input_data):

        h = F.relu(self.in_layer(input_data))

        for layer in self.hidden_layers:
            h = F.relu(layer(h))

        h = self.dropout(h)
        return self.out_layer(h)


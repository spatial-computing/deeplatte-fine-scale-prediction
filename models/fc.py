from torch import nn
import torch.nn.functional as F


class FC(nn.Module):

    def __init__(self, in_dim, h_dims, out_dim, **kwargs):
        """
        At least one hidden layers
        
        Params:
            in_dim: number of dimensions of input layer
            h_dims: a list of number of dimensions of hidden layers
            out_dim: number of dimensions of output layer
        """
 
        super(FC, self).__init__()
    
        self.in_dim = in_dim
        self.h_dims = h_dims
        self.out_dim = out_dim
        self.p_dropout = kwargs.get('p_dropout', 0.0)
            
        # input layer
        self.in_layer = nn.Linear(self.in_dim, self.h_dims[0])
        
        # hidden layers
        self.h_layers = nn.ModuleList()
        for i in range(len(self.h_dims) - 1):
            self.h_layers.append(nn.Linear(self.h_dims[k], self.h_dims[i + 1]))

        # output layer
        self.dropout = nn.Dropout(p=self.p_dropout)
        self.out_layer = nn.Linear(self.h_dims[-1], self.out_dim)

    def forward(self, input_data):

        h = self.in_layer(input_data)
        for l in self.h_layers:
            h = F.relu(l(h))

        h = self.dropout(h)
        return self.out_layer(h)


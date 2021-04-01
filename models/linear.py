from torch import nn
import torch
from torch.nn.utils.prune import l1_unstructured


class DiagPruneLinear(nn.Module):
    """ a diagonal linear layer with weight pruning """
    
    def __init__(self, in_features, **kwargs):
        """
        params:
            in_features (int): size of input sample
        """
        
        super(DiagPruneLinear, self).__init__()

        self.device = kwargs.get('device', 'cpu')

        self.linear_layer = nn.Linear(in_features, in_features, bias=False)

        # create a mask that has 0s in diagonal
        self.mask = torch.ones([in_features, in_features], dtype=torch.bool).to(self.device)
        mask_indices = [[range(in_features)], [range(in_features)]]  # [list, list]
        self.mask[mask_indices] = 0
        
        def backward_hook(grad):
            out = grad.clone()  # clone due to not being allowed to modify in-place gradients
            out[self.mask] = 0
            return out
 
        self.linear_layer.weight.data[self.mask] = 0  # only keep the diagonal weight
        self.linear_layer.weight.register_hook(backward_hook)  # hook to zero out bad gradients

        # register a prune on the weights by l1 norm
        l1_unstructured(self.linear_layer, name='weight', amount=0.996)

    def l1_loss(self):
        return self.linear_layer.weight.abs().sum()

    def forward(self, input_data):
        return self.linear_layer(input_data)


class Stack2Linear(nn.Module):

    def __init__(self, in_features, h_features, out_features, **kwargs):
        """
        params:
            in_features (int): size of input sample
            h_features (int): size of hidden state
            out_features (int): size of output sample
        """

        super(Stack2Linear, self).__init__()

        p = kwargs.get('dropout', 0.5)

        self.lin1 = nn.Sequential(nn.Linear(in_features, h_features), nn.ReLU(), nn.Dropout(p=p))
        self.lin2 = nn.Linear(h_features, out_features)

    def forward(self, input_data):
        output = self.lin1(input_data)
        output = self.lin2(output)
        return output

from torch import nn
import torch


class MaskNet(nn.Module):
    """ a linear layer with a given mask """
    
    def __init__(self, in_dim, out_dim, mask_indices, **kwargs):
        """
        Params:
            in_dim: number of channels of input tensor
            out_dim: number of channels of output tensor
            mask_indices: [list, list]
                containing indices for the mask on weights
        """
        
        super(MaskNet, self).__init__()
 
        self.device = kwargs.get('device', 'cpu')
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=False)
        
        # create a mask that has 0s in diagonal
        self.mask = torch.ones([out_dim, in_dim], dtype=torch.bool).to(self.device)
        self.mask[mask_indices] = 0
        
        def backward_hook(grad):
            out = grad.clone()  # clone due to not being allowed to modify in-place gradients
            out[self.mask] = 0
            return out
 
        self.linear_layer.weight.data[self.mask] = 0  # only keep the weight where mask == 0
        self.linear_layer.weight.register_hook(backward_hook)  # hook to zero out bad gradients
 
    def forward(self, input_data):
        return self.linear_layer(input_data)

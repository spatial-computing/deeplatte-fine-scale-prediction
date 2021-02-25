from torch import nn
import torch


class MaskNet(nn.Module):
    
    def __init__(self, in_dim, out_dim, indices_mask, **kwargs):
        """
        Params:
            in_dim: int
                Number of channels of input tensor
            out_dim: int
                Number of channels of output tensor
            indices_mask: [list, list]
                containing indices for dimensions 0 and 1, used to create the mask
        """
        
        super(MaskNet, self).__init__()
 
        self.device = kwargs.get('device', 'cpu')

        def backward_hook(grad):
            # clone due to not being allowed to modify in-place gradients
            out = grad.clone()
            out[self.mask] = 0
            return out
 
        # output
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=False)
        
        # create a mask for zero out useless weight
        self.mask = torch.ones([out_dim, in_dim]).byte().to(self.device)
        self.mask[indices_mask] = 0
        
        # only keep the weight where mask == 0
        self.linear_layer.weight.data[self.mask] = 0  # zero out bad weights
        self.linear_layer.weight.register_hook(backward_hook)  # hook to zero out bad gradients
 
    def forward(self, input_data):
        return self.linear_layer(input_data)

import torch.nn.functional as F
from torch import nn
import torch


class FeatureAttention(nn.Module):
    """ a linear layer with a given mask """
    
    def __init__(self, in_dim, out_dim, **kwargs):
        """
        Params:
            in_dim: number of channels of input tensor
            out_dim: number of channels of output tensor
            mask_indices: [list, list]
                containing indices for the mask on weights
        """
        
        super(FeatureAttention, self).__init__()
 
        self.device = kwargs.get('device', 'cpu')
        self.attention_layer = nn.Linear(in_dim, out_dim, bias=True)
        
    def forward(self, input_data):
        a = F.sigmoid(self.attention_layer(input_data))
        a = F.softmax(a)
        return a * input_data

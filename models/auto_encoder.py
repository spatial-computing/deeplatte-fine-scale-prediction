from torch import nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """ a Vallina auto-encoder """
    
    def __init__(self, in_dim, en_h_dims, de_h_dims, **kwargs):
        """
        At least one encoded hidden layers and at least one decoded hidden layers
        
        Params:
            in_dim: the number of dimensions of the input data
            en_h_dims: encoder layers
            de_h_dims: decoder layers 
            Assert en_h_dims[-1] == de_h_dims[0]

        """

        super(AutoEncoder, self).__init__()

        self.en_h_dims = [in_dim] + en_h_dims
        self.de_h_dims = de_h_dims + [in_dim]
        self.p_dropout = kwargs.get('p_dropout', 0.1)

        # encoding layers
        self.encoder = nn.ModuleList()
        for i in range(len(self.en_h_dims) - 1):
            self.encoder.append(nn.Linear(self.en_h_dims[i], self.en_h_dims[i + 1]))

        # decoding layers
        self.decoder = nn.ModuleList()
        for i in range(len(self.de_h_dims) - 1):
            self.decoder.append(nn.Linear(self.de_h_dims[i], self.de_h_dims[i + 1]))
        
        self.dropout = nn.Dropout(p=self.p_dropout)

    def forward(self, input_data):

        en = input_data

        for i, l in enumerate(self.encoder):
            en = F.relu(l(en))
#             en = l(en)
#             if i < len(self.encoder) - 1:
#                 en = F.relu(en)

        de = en
        for i, l in enumerate(self.decoder):
            de = l(de)
            if i < len(self.decoder) - 1:
                de = F.relu(de)
                
        return en, de

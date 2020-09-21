from torch import nn
import torch.nn.functional as F


# a simple Auto-Encoder model

class AutoEncoder(nn.Module):

    def __init__(self, in_dim, en_h_dims, de_h_dims):

        super(AutoEncoder, self).__init__()

        self.input_dim = in_dim
        self.encoder_hidden_dims = en_h_dims
        self.decoder_hidden_dims = de_h_dims

        # input layer
        self.in_layer = nn.Linear(self.input_dim, self.encoder_hidden_dims[0])

        # encoding layers
        self.encoder = nn.ModuleList()
        for k in range(len(self.encoder_hidden_dims) - 1):
            self.encoder.append(nn.Linear(self.encoder_hidden_dims[k], self.encoder_hidden_dims[k + 1]))

        # decoding layers
        self.decoder = nn.ModuleList()
        for k in range(len(self.decoder_hidden_dims) - 1):
            self.decoder.append(nn.Linear(self.decoder_hidden_dims[k], self.decoder_hidden_dims[k + 1]))

        # output layer
        self.out_layer = nn.Linear(self.decoder_hidden_dims[-1], self.input_dim)

    def forward(self, input_data):

        encoded = F.relu(self.in_layer(input_data))

        for layer in self.encoder:
            encoded = F.relu(layer(encoded))

        decoded = encoded

        for layer in self.decoder:
            decoded = F.relu(layer(decoded))

        decoded = self.out_layer(decoded)
        return encoded, decoded

from torch import nn


class AutoEncoder(nn.Module):
    """ a vallina auto-encoder """
    
    def __init__(self, in_features, en_features, de_features, **kwargs):
        """
        at least one encoded hidden layers and at least one decoded hidden layers
        
        params:
            in_features: the number of dimensions of the input data
            en_features: encoder layers
            de_features: decoder layers
            assert en_features[-1] == de_features[0]

        """

        super(AutoEncoder, self).__init__()

        en_features = [in_features] + en_features
        de_features = de_features + [in_features]
        p = kwargs.get('p', 0.5)

        # encoding layers
        encoder = []
        for i in range(len(en_features) - 1):
            encoder.append(nn.Linear(en_features[i], en_features[i + 1]))
            encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)

        # decoding layers
        decoder = []
        for i in range(len(de_features) - 1):
            decoder.append(nn.Linear(de_features[i], de_features[i + 1]))
            if i == len(de_features) - 1:
                decoder.append(nn.Sigmoid())
            else:
                decoder.append(nn.ReLU())
                decoder.append(nn.Dropout(p=p))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, input_data):

        en = self.encoder(input_data)
        de = self.decoder(en)
        return en, de

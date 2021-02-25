from torch import nn
import torch


class SpatialLossFunc(nn.Module):

    def __init__(self, sp_neighbor):
        """
        Params:
            sp_neighbor: number of neighbors
        """
        
        super(SpatialLossFunc, self).__init__()
        
        self.sp_neighbor = sp_neighbor

    def forward(self, input_data):

        loss = 0.
        t, _, h, w = input_data.shape

        for i in range(-self.sp_neighbor, self.sp_neighbor + 1):
            for j in range(-self.sp_neighbor, self.sp_neighbor + 1):
                weight = (i * i + j * j) ** 0.5
                if i >= 0 and j >= 0 and weight != 0:
                    loss += torch.sum((input_data[..., i:, j:] - input_data[..., : h - i, : w - j]) ** 2) / weight
                elif i >= 0 and j < 0:
                    loss += torch.sum((input_data[..., i:, :j] - input_data[..., : h - i, -j:]) ** 2) / weight
                elif i < 0 and j >= 0:
                    loss += torch.sum((input_data[..., :i, j:] - input_data[..., -i:, : w - j]) ** 2) / weight
                elif i < 0 and j < 0:
                    loss += torch.sum((input_data[..., :i, :j] - input_data[..., -i:, -j:]) ** 2) / weight
                else:
                    pass

        return loss / t / h / w

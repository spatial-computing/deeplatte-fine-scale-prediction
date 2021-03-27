from torch import nn
import torch


class OneStepSpatialLoss(nn.Module):

    def __init__(self):

        super(OneStepSpatialLoss, self).__init__()
        self.mse_loss_func = nn.MSELoss()
        self.loss = 0.

    def forward(self, input_data):
        self.loss += self.mse_loss_func(input_data[..., 1:, 1:], input_data[..., :-1, :-1])
        self.loss += self.mse_loss_func(input_data[..., 1:, :], input_data[..., :-1, :])
        self.loss += self.mse_loss_func(input_data[..., :, 1:], input_data[..., :, :-1])
        return self.loss


class SpatialLoss(nn.Module):

    def __init__(self, sp_neighbor):
        """ sp_neighbor: number of spatial neighbors """

        super(SpatialLoss, self).__init__()
        self.sp_neighbor = sp_neighbor
        self.mse_loss_func = nn.MSELoss()
        self.loss = 0.

    def forward(self, input_data):

        t, _, h, w = input_data.shape

        for i in range(-self.sp_neighbor, self.sp_neighbor + 1):
            for j in range(-self.sp_neighbor, self.sp_neighbor + 1):
                weight = (i * i + j * j) ** 0.5
                if i >= 0 and j >= 0 and weight != 0:
                    self.loss += self.mse_loss_func(input_data[..., i:, j:] - input_data[..., : h - i, : w - j]) / weight
                elif i >= 0 and j < 0:
                    self.loss += self.mse_loss_func(input_data[..., i:, :j] - input_data[..., : h - i, -j:]) / weight
                elif i < 0 and j >= 0:
                    self.loss += self.mse_loss_func(input_data[..., :i, j:] - input_data[..., -i:, : w - j]) / weight
                elif i < 0 and j < 0:
                    self.loss += self.mse_loss_func(input_data[..., :i, :j] - input_data[..., -i:, -j:]) / weight
                else:
                    pass

        return self.loss


class TemporalLoss(nn.Module):

    def __init__(self, tp_neighbor):
        """ sp_neighbor: number of spatial neighbors """

        super(TemporalLoss, self).__init__()
        self.tp_neighbor = tp_neighbor
        self.mse_loss_func = nn.MSELoss()
        self.loss = 0.

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

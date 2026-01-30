import numpy as np
import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['Loss_PhyHysLSTM']
class Loss_PhyHysLSTM(nn.Module):
    def __init__(self,dt):
        super(Loss_PhyHysLSTM, self).__init__()
        self.mse = nn.MSELoss()
        self.dt =dt
    def forward(self, input_data, target_all, pred_all):

        disp_true, vel_true, acc_true = target_all[..., 0], target_all[..., 1], target_all[..., 2]
        disp_pred, vel_pred, acc_pred = pred_all[..., 0], pred_all[..., 1], pred_all[..., 2]
        data_loss_u = self.mse(disp_pred, disp_true)
        data_loss_v = self.mse(vel_pred, vel_true)
        data_loss_a = self.mse(acc_pred, acc_true)
        loss_fit = (data_loss_u +data_loss_v +data_loss_a) / 3
        return loss_fit


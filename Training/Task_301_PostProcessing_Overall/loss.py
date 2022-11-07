# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import setting


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, pred, gt):
        gt = gt[0]
        pred = pred[0]

        gt = gt.float().detach().unsqueeze(1)
        pred = torch.sigmoid(pred)
        loss = self.bce_loss(pred, gt)
        return loss

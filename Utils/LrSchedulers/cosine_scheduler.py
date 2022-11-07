# -*- encoding: utf-8 -*-
import torch
import numpy as np


class CosineScheduler:
    def __init__(self, T_max, eta_min=1e-8, last_epoch=-1):
        self.type = 'CosineScheduler'
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch

    def get_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.T_max,
                                                               eta_min=self.eta_min,
                                                               last_epoch=self.last_epoch
                                                               )
        return scheduler


class CosineWithWarmupScheduler:
    def __init__(self, T_max, warmup_iter, eta_min=1e-8):
        self.type = 'CosineWithWarmupScheduler'
        self.T_max = T_max
        self.warmup_iter = warmup_iter
        self.eta_min = eta_min

    def get_scheduler(self, optimizer):
        def lambda_func(x):
            if x <= self.warmup_iter:
                return (1.0 - self.eta_min) * x / self.warmup_iter + self.eta_min
            else:
                cos_v = np.cos((x - self.warmup_iter) * np.pi / self.T_max)
                cos_v = 0.5 * (cos_v + 1.) * (1 - self.eta_min) + self.eta_min

                return cos_v
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
        return scheduler

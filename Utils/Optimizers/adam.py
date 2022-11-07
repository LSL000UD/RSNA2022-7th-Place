# -*- encoding: utf-8 -*-
import torch


class Adam:
    def __init__(self, lrs, weight_decay=3e-5, betas=(0.9, 0.999), eps=1e-08, amsgrad=True):
        self.type = 'Adam'
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad
        self.weight_decay = weight_decay
        self.lrs = lrs

        if not isinstance(self.lrs, list):
            self.lrs = [self.lrs]

    def get_optimizer(self, list_param_groups):
        if not isinstance(list_param_groups, list):
            list_param_groups = list(list_param_groups)

        # Sometimes we need set different learning rates for "encoder" and "decoder" separately
        list_param_lr_groups = []
        for i in range(len(list_param_groups)):
            list_param_lr_groups.append({'params': list_param_groups[i], 'lr': self.lrs[i]})

        optimizer = torch.optim.Adam(list_param_lr_groups,
                                     weight_decay=self.weight_decay,
                                     betas=self.betas,
                                     eps=self.eps,
                                     amsgrad=self.amsgrad)
        return optimizer

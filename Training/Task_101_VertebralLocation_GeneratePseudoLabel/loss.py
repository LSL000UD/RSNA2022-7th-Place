import warnings

import torch.nn as nn

from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2



class CustomLoss(nn.Module):
    def __init__(self, ds_loss_weights):
        super(CustomLoss, self).__init__()
        print(f"==> Using custom loss from {__file__}")

        self.ds_loss_weights = ds_loss_weights

        base_loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        self.loss = MultipleOutputLoss2(base_loss, ds_loss_weights)

        self.first_iter = True

    def forward(self, net_output, target):
        if self.first_iter:
            warnings.warn(f"==> {net_output[0].shape}")
            self.first_iter = False
        return self.loss(net_output, target)

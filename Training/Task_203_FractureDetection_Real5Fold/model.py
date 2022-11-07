import warnings

import torch.nn as nn

from nnunet.network_architecture.generic_UNet import Generic_UNet


class CustomModel(Generic_UNet):
    def __init__(self, *args, **kwargs):
        # Set input channels to 1
        args = (1, ) + args[1:]

        super(CustomModel, self).__init__(*args, **kwargs)
        print(f"==> Using custom model from {__file__}")

        self.first_iter = True

    def forward(self, x):
        if self.first_iter:
            warnings.warn(f"==> {x.shape}")
            self.first_iter = False
        return super(CustomModel, self).forward(x)

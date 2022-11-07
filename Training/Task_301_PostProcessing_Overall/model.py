import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(Encoder, self).__init__()

        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_2 = nn.Sequential(
            SingleConv(list_ch[1], list_ch[2], kernel_size=3, stride=2, padding=(1, 1, 1)),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=(1, 1, 1)),
        )
        self.encoder_3 = nn.Sequential(
            SingleConv(list_ch[2], list_ch[3], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
        )
        self.encoder_4 = nn.Sequential(
            SingleConv(list_ch[3], list_ch[4], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)

        return out_encoder_4


class Decoder(nn.Module):
    def __init__(self, out_ch, list_ch):
        super(Decoder, self).__init__()
        self.cls = nn.Linear(list_ch[-1], out_ch)

    def forward(self, out_encoder):
        out = out_encoder
        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(out.size(0), -1)
        out = self.cls(out)

        return [out]


class Model(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch, random_init=True):
        super(Model, self).__init__()
        self.encoder = Encoder(in_ch, list_ch)
        self.decoder = Decoder(out_ch, list_ch)

        self.info = str(list_ch)

        # init
        if random_init:
            self.initialize()

    def initialize(self):
        print('==> Random init weights using MSRA')
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.)

    def forward(self, x, cur_iter=None):
        if isinstance(x, list):
            x = x[0]
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)  # is a list

        return out_decoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is SiLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=True):
        super().__init__()
        self.dconv = BN_Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels,
                               activation=activation)
        self.pconv = BN_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, groups=1, activation=activation)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)
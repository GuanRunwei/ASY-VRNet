import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format
import math

from backbone.vision.context_cluster import ClusterBlock
from backbone.vision.context_cluster import coc_medium


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.dconv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               stride=stride, groups=in_channels, padding=padding, dilation=dilation, bias=bias)
        self.pconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, groups=1,
                               bias=bias)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="relu", ds_conv=False):
        super().__init__()
        pad         = (ksize - 1) // 2
        if ds_conv is False:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad,
                                  groups=groups, bias=bias)
        else:
            self.conv = DWConv(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad,
                                  groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            ClusterBlock(dim=in_channels),
            BaseConv(in_channels, out_channels, 1, 1, act='relu'),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x


class CoC_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="relu"):
        super(CoC_Conv, self).__init__()

        self.coc = ClusterBlock(dim=in_channels)
        self.conv_att = BaseConv(in_channels, out_channels, ksize=ksize, stride=stride, act=act)

    def forward(self, x):
        x = self.coc(x)
        x = self.conv_att(x)
        return x


# -----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            DWConv(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            DWConv(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            DWConv(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.avg_pool(x)
        # print("average pool shape:", output.shape)
        output = self.conv(output.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        output = self.sigmoid(output)

        return x * output.expand_as(x)


class FpnTiny(nn.Module):
    def __init__(self, num_seg_class, depth=1.0, width=1.0, in_features=("dark2", "dark3", "dark4", "dark5"),
                 in_channels=[128, 320, 512], stage2_channel=64, aspp_channel=1024):
        super(FpnTiny, self).__init__()

        Conv = CoC_Conv

        self.backbone = coc_medium(pretrained=False)
        self.aspp = ASPP(dim_in=in_channels[-1], dim_out=in_channels[-1]*2)
        self.in_features = in_features
        self.num_seg_class = num_seg_class


    def forward(self, x):




if __name__ == '__main__':
    input_map = torch.randn((1, 512, 20, 20)).cuda()
    aspp = ASPP(dim_in=512, dim_out=1024).cuda()
    macs, params = profile(aspp, inputs=(input_map,))
    macs, params = clever_format([macs, params], "%.3f")
    print("params:", params)
    print("macs:", macs)
    # yoloneck = FpnTiny(num_seg_class=21).cuda()
    # output, seghead = yoloneck(input_map)
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)
    # print(seghead.shape)
    # macs, params = profile(yoloneck, inputs=(input_map,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print("params:", params)
    # print("macs:", macs)





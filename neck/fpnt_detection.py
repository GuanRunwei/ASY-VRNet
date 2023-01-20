import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format
import math

from backbone.vision.context_cluster import ClusterBlock
from backbone.vision.context_cluster import coc_medium, coc_small
from backbone.attention_modules.shuffle_attention import ShuffleAttention


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
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="relu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="relu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
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
        super(DW_CoC_Conv, self).__init__()

        self.coc = ClusterBlock(dim=in_channels)
        self.conv_att = BaseConv(in_channels, out_channels, ksize=ksize, stride=stride, act=act)

    def forward(self, x):
        x = self.coc(x)
        x = self.conv_att(x)
        return x


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
    def __init__(self, num_seg_class, depth=1.0, width=1.0, in_features=("dark2", "dark3", "dark4", "dark5"), in_channels=[128, 320, 512],
                 stage2_channel=80, stage1_channel=64,
                 depthwise=False, act="silu", is_attention=2):
        super(FpnTiny, self).__init__()

        Conv = CoC_Conv

        self.backbone = coc_small(pretrained=False)
        self.in_features = in_features
        self.num_seg_class = num_seg_class

        # p5 out
        self.conv_p5 = Conv(in_channels=int(in_channels[2]*width), out_channels=int(in_channels[2]*width),
                            ksize=1, stride=1)

        # c4 -> p4
        self.conv_p4 = Conv(in_channels=int(in_channels[1]*width), out_channels=int(in_channels[1]*width),
                            ksize=1, stride=1)

        # c3 -> p3
        self.conv_p3 = Conv(in_channels=int(in_channels[0] * width), out_channels=int(in_channels[0] * width),
                            ksize=1, stride=1)

        # c2 -> p2
        self.conv_p2 = Conv(in_channels=int(stage2_channel * width), out_channels=int(stage2_channel * width),
                            ksize=1, stride=1)

        # 20 * 20 * 1024 -> 40 * 40 * 512
        self.upsampling_p5 = Upsample(in_channels=int(in_channels[2]*width), out_channels=int(in_channels[1]*width))

        # 40 * 40 * 512 -> 80 * 80 * 256
        self.upsampling_p4 = Upsample(in_channels=int(in_channels[1]*width), out_channels=int(in_channels[0]*width))

        # 80 * 80 * 256 -> 160, 160, 128
        self.upsampling_p3 = Upsample(in_channels=int(in_channels[0]*width), out_channels=int(stage2_channel*width))

        # 120, 120, 80 -> 240. 240. 40
        self.upsampling_seg2 = Upsample(in_channels=int(in_channels[0] * width),
                                        out_channels=int(stage2_channel * width), scale=4)

        self.conv_seg = nn.Sequential(
            Conv(in_channels=int(stage2_channel * width), out_channels=int(stage2_channel * width), ksize=3, stride=1,
                 act='relu'),
            Conv(in_channels=int(stage2_channel * width), out_channels=int(stage2_channel * width), ksize=1, stride=1,
                 act='relu'),
            Conv(in_channels=int(stage2_channel * width), out_channels=num_seg_class, ksize=1, stride=1, act='relu'),
        )

        self.eca_seg1 = eca_block(channel=int(in_channels[0] * width))
        self.eca_seg2 = eca_block(channel=int(stage2_channel * width))

    def forward(self, input):
        out_features = self.backbone.forward(input)
        [feat0, feat1, feat2, feat3] = [out_features[f] for f in range(len(out_features))]

        # ========================================  检测  ============================================= #
        #  P5 输出 20*20
        P5_out = self.conv_p5(feat3)

        #  P4 输出 40*40
        P4 = self.conv_p4(feat2)
        P5_4 = self.upsampling_p5(feat3)
        P4_out = P4 + P5_4

        #  P3 输出 80*80
        P3 = self.conv_p3(feat1)
        P4_3 = self.upsampling_p4(feat2)
        P3_out = P3 + P4_3

        # ============================================================================================ #
        detect_p_out = (P3_out, P4_out, P5_out)

        return detect_p_out


if __name__ == '__main__':
    input_map = torch.randn((1, 3, 512, 512)).cuda()
    yoloneck = FpnTiny(num_seg_class=21).cuda()
    output, seghead = yoloneck(input_map)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(seghead.shape)
    macs, params = profile(yoloneck, inputs=(input_map,))
    macs, params = clever_format([macs, params], "%.3f")
    print("params:", params)
    print("macs:", macs)





import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format
import math
from backbone.attention_modules.eca import eca_block
from backbone.attention_modules.shuffle_attention import ShuffleAttention
from backbone.conv_utils.normal_conv import DWConv, BaseConv
from backbone.vision.context_cluster import ClusterBlock
from backbone.fusion.vr_coc import coc_medium


class CoCUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()

        self.upsample = nn.Sequential(
            BaseConv(in_channels, out_channels, 1, 1, act='relu', ds_conv=True),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x


class CoC_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="relu", ds_conv=True):
        super(CoC_Conv, self).__init__()

        self.coc = ClusterBlock(dim=in_channels)
        self.conv_att = BaseConv(in_channels, out_channels, ksize=ksize, stride=stride, act=act, ds_conv=ds_conv)

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


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


def shuffle_channels(x, groups=2):
    """Channel Shuffle"""

    batch_size, channels, h, w = x.data.size()
    if channels % groups:
        return x
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, h, w)
    return x


class CoCFpnDual(nn.Module):
    def __init__(self, num_seg_class=5, depth=1.0, width=1.0, in_features=("dark2", "dark3", "dark4", "dark5"),
                 in_channels=[64, 128, 320, 512], aspp_channel=1024):
        super().__init__()

        Conv = CoC_Conv

        self.backbone = coc_medium(pretrained=False, width=width)
        self.in_features = in_features
        self.num_seg_class = num_seg_class
        in_channels = [int(item*width) for item in in_channels]

        self.aspp = ASPP(dim_in=in_channels[-1], dim_out=in_channels[-1])

        # ================================= segmentation modules =================================== #
        # ----------------------- 20*20*512 -> 40*40*320 -> 40*40*640 ------------------------ #
        self.upsample5_4 = CoCUpsample(in_channels=in_channels[-1], out_channels=in_channels[-2])
        self.sc_attn_seg4 = ShuffleAttention(channel=in_channels[-2]*2)
        # ------------------------------------------------------------------------------------- #

        # ----------------------- 40*40*640 -> 80*80*128 -> 80*80*256 ------------------------ #
        self.upsample4_3 = CoCUpsample(in_channels=in_channels[-2]*2, out_channels=in_channels[-3])
        self.sc_attn_seg3 = ShuffleAttention(channel=in_channels[-3] * 2)
        # ------------------------------------------------------------------------------------ #

        # ----------------------- 80*80*256 -> 160*160*64 -> 160*160*128 ------------------------ #
        self.upsample3_2 = CoCUpsample(in_channels=in_channels[-3] * 2, out_channels=in_channels[0])
        self.sc_attn_seg2 = ShuffleAttention(channel=in_channels[0] * 2)
        # ------------------------------------------------------------------------------------ #

        # ----------------------- 80*80*256 -> 160*160*64 -> 160*160*128 ------------------------ #
        self.upsample2_0 = CoCUpsample(in_channels=in_channels[0] * 2, out_channels=1, scale=4)
        # ------------------------------------------------------------------------------------ #
        # ========================================================================================== #

        # ================================= detection modules ====================================== #
        # ----------------------- 20*20*512 -> 20*20*512 ----------------------- #
        self.p5_out_det = CoC_Conv(in_channels=in_channels[-1], out_channels=in_channels[-1])
        # ----------------------------------------------------------------------- #

        # ----------------------- 20*20*512 -> 40*40*320 -> 40*40*640 -> 40*40*320 ------------------------ #
        self.p5_4_det = CoCUpsample(in_channels=in_channels[-1], out_channels=in_channels[-2])
        self.p4_out_det = CoC_Conv(in_channels=in_channels[-2]*2, out_channels=in_channels[-2])
        # ------------------------------------------------------------------------------------------------- #

        # ----------------------- 40*40*320 -> 80*80*128 -> 80*80*256 -> 80*80*128 ------------------------ #
        self.p4_3_det = CoCUpsample(in_channels=in_channels[-2], out_channels=in_channels[-3])
        self.p3_out_det = CoC_Conv(in_channels=in_channels[-3]*2, out_channels=in_channels[-3])
        # ------------------------------------------------------------------------------------------------- #
        # ========================================================================================== #

    def forward(self, x, x_radar):

        x_out, x_radar_out = self.backbone(x, x_radar)

        x_stage2, x_stage3, x_stage4, x_stage5 = x_out
        x_stage5 = self.aspp(x_stage5)

        x_radar_stage2, x_radar_stage3, x_radar_stage4, x_radar_stage5 = x_radar_out

        # ---------------------------- segmentation ------------------------------- #
        x_stage5_4 = self.upsample5_4(x_stage5)
        x_stage4_concat_5 = torch.cat([x_stage4, x_stage5_4], dim=1)
        x_stage4_concat_5 = shuffle_channels(x_stage4_concat_5)
        x_stage4_concat_5 = self.sc_attn_seg4(x_stage4_concat_5)

        x_stage4_3 = self.upsample4_3(x_stage4_concat_5)
        x_stage3_concat_4 = torch.cat([x_stage4_3, x_stage3], dim=1)
        x_stage3_concat_4 = shuffle_channels(x_stage3_concat_4)
        x_stage3_concat_4 = self.sc_attn_seg3(x_stage3_concat_4)

        x_stage3_2 = self.upsample3_2(x_stage3_concat_4)
        x_stage2_concat_3 = torch.cat([x_stage3_2, x_stage2], dim=1)
        x_stage2_concat_3 = shuffle_channels(x_stage2_concat_3)
        x_stage2_concat_3 = self.sc_attn_seg2(x_stage2_concat_3)

        x_segmentation_out = self.upsample2_0(x_stage2_concat_3)
        # ------------------------------------------------------------------------ #

        # ----------------------------- detection -------------------------------- #
        p5_out = self.p5_out_det(x_radar_stage5)

        p5_4_upsample = self.p5_4_det(p5_out)
        p4_concat_5 = torch.cat([x_radar_stage4, p5_4_upsample], dim=1)
        p4_out = self.p4_out_det(p4_concat_5)

        p4_3_upsample = self.p4_3_det(p4_out)
        p3_concat_4 = torch.cat([x_radar_stage3, p4_3_upsample], dim=1)
        p3_out = self.p3_out_det(p3_concat_4)
        # ------------------------------------------------------------------------ #

        return x_segmentation_out, [p3_out, p4_out, p5_out]


if __name__ == '__main__':
    # input_map = torch.randn((1, 512, 20, 20)).cuda()
    # aspp = ASPP(dim_in=512, dim_out=1024).cuda()
    model = CoCFpnDual(width=1.0)
    model.eval()
    input = torch.rand(1, 3, 640, 640)
    input_radar = torch.rand(1, 4, 640, 640)
    output = model(input, input_radar)
    macs, params = profile(model, inputs=(input, input_radar))
    macs, params = clever_format([macs, params], "%.3f")
    print("params:", params)
    print("macs:", macs)
    print(output[0].shape)
    print(output[1][0].shape)
    print(output[1][1].shape)
    print(output[1][2].shape)
    # model = SpatialPyramidPooling()
    # input = torch.rand(1, 512, 20, 20)
    # output = model(input)
    # print(output.shape)





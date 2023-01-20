from backbone.radar.context_cluster import coc_small as coc_small_radar, coc_medium as coc_medium_radar, coc_tiny2 as coc_tiny2_radar
from backbone.vision.context_cluster import coc_small, coc_medium, coc_tiny2
from backbone.vision.context_cluster import ClusterBlock
from backbone.attention_modules.shuffle_attention import ShuffleAttention
from neck.fpnt_segmentation import BaseConv, DWConv
from neck.fpnt_segmentation import eca_block
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchinfo import summary


def data_normal(origin_data):
    d_min = origin_data.min()
    if d_min < 0:
        origin_data += torch.abs(d_min)
        d_min = origin_data.min()
    d_max = origin_data.max()
    dst = d_max - d_min
    norm_data = (origin_data - d_min).true_divide(dst)
    return norm_data


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


class VRCoC(nn.Module):
    def __init__(self, version='s', backbone_pretrain=False):
        super(VRCoC, self).__init__()

        # ========================= stage 1 -> initialization =================================== #
        self.vision_initial = ClusterBlock(dim=3)
        self.radar_initial = ClusterBlock(dim=4)
        # ------------------------- vision to radar attention ----------------------------------- #
        self.vision_to_radar_attn1 = BaseConv(in_channels=3, out_channels=3, ksize=1, stride=1)
        # --------------------------------------------------------------------------------------- #

        # ------------------------- radar to vision projection ---------------------------------- #
        self.radar_to_vision_projection = BaseConv(in_channels=4, out_channels=1, ksize=1, stride=1)
        # --------------------------------------------------------------------------------------- #
        # ======================================================================================= #

        # ============================ CoC backbone initialization ============================== #
        if version == 'tiny':
            self.radar_backbone = coc_tiny2_radar(pretrained=backbone_pretrain)
            self.vision_backbone = coc_tiny2(pretrained=backbone_pretrain)
            self.in_channels = [32, 64, 196, 320]
        elif version == 's':
            self.radar_backbone = coc_small_radar(pretrained=backbone_pretrain)
            self.vision_backbone = coc_small(pretrained=backbone_pretrain)
            self.in_channels = [64, 128, 320, 512]
        elif version == 'm':
            self.radar_backbone = coc_medium_radar(pretrained=backbone_pretrain)
            self.vision_backbone = coc_medium(pretrained=backbone_pretrain)
            self.in_channels = [64, 128, 320, 512]
        # ======================================================================================= #

        # =================================== stage 2 ========================================= #
        # -------------------------- stage 2 -> vision branch -------------------------- #
        self.radar_to_vision_projection2 = BaseConv(in_channels=self.in_channels[0], out_channels=1, ksize=1, stride=1)

        # -------------------------- stage 2 -> radar branch --------------------------- #
        self.vision_to_radar_attn2 = ShuffleAttention(channel=self.in_channels[0], G=4)
        self.radar_branch_channel_attn1 = eca_block(channel=self.in_channels[0]*2)
        self.inverse_projection1 = DWConv(in_channels=self.in_channels[0]*2, out_channels=self.in_channels[0], ksize=1)
        # ------------------------------------------------------------------------------ #
        # ===================================================================================== #

        # =================================== stage 3  ======================================== #
        # -------------------------- stage 3 -> vision branch -------------------------- #
        self.radar_to_vision_projection3 = BaseConv(in_channels=self.in_channels[1], out_channels=1, ksize=1, stride=1)
        # ------------------------------------------------------------------------------ #

        # -------------------------- stage 3 -> radar branch --------------------------- #
        self.vision_to_radar_attn3 = ShuffleAttention(channel=self.in_channels[1], G=4)
        self.radar_branch_channel_attn2 = eca_block(channel=self.in_channels[1] * 2)
        self.inverse_projection2 = DWConv(in_channels=self.in_channels[1] * 2, out_channels=self.in_channels[1], ksize=1)
        # ------------------------------------------------------------------------------ #
        # ===================================================================================== #

        # =================================== stage 4  ======================================== #
        # -------------------------- stage 4 -> vision branch -------------------------- #
        self.radar_to_vision_projection4 = BaseConv(in_channels=self.in_channels[2], out_channels=1, ksize=1, stride=1)
        # ------------------------------------------------------------------------------ #

        # -------------------------- stage 4 -> radar branch --------------------------- #
        self.vision_to_radar_attn4 = ShuffleAttention(channel=self.in_channels[2], G=4)
        self.radar_branch_channel_attn3 = eca_block(channel=self.in_channels[2] * 2)
        self.inverse_projection3 = DWConv(in_channels=self.in_channels[2] * 2, out_channels=self.in_channels[2], ksize=1)
        # ------------------------------------------------------------------------------ #
        # ===================================================================================== #

        # =================================== stage 5  ======================================== #
        # -------------------------- stage 5 -> vision branch -------------------------- #
        self.radar_to_vision_projection5 = BaseConv(in_channels=self.in_channels[3], out_channels=1, ksize=1, stride=1)
        # ------------------------------------------------------------------------------ #

        # -------------------------- stage 5 -> radar branch --------------------------- #
        self.vision_to_radar_attn5 = ShuffleAttention(channel=self.in_channels[3], G=4)
        self.radar_branch_channel_attn4 = eca_block(channel=self.in_channels[3] * 2)
        self.inverse_projection4 = DWConv(in_channels=self.in_channels[3] * 2, out_channels=self.in_channels[3], ksize=1)
        # ------------------------------------------------------------------------------ #
        # ===================================================================================== #

    def forward(self, image, radar):
        image_feat = self.vision_initial(image)
        radar_feat = self.radar_initial(radar)

        # ===================== the first asymmetric fair fusion ===================== #
        image_feat_prepared_to_radar = self.vision_to_radar_attn1(image_feat)
        radar_feat_prepared_to_image = self.radar_to_vision_projection(radar_feat)

        # --------------------- image after enhancement 1 3*560*560 ------------------ #
        image_feat1 = (1 + data_normal(radar_feat_prepared_to_image)) * image_feat
        # ---------------------------------------------------------------------------- #
        # --------------------- radar after enhancement 2 7*560*560------------------- #
        radar_feat1 = torch.cat([radar_feat, image_feat_prepared_to_radar], dim=1)
        # radar_feat1 = shuffle_channels(radar_feat1, 2)
        # ---------------------------------------------------------------------------- #

        # ============================================================================ #

        # ============================== CoC backbone ================================ #
        image_features = self.vision_backbone(image_feat1)
        radar_features = self.radar_backbone(radar_feat1)

        # --------------------------- CoC features list ------------------------------ #
        image_feat2 = image_features[0]
        radar_feat2 = radar_features[0]

        image_feat3 = image_features[1]
        radar_feat3 = radar_features[1]

        image_feat4 = image_features[2]
        radar_feat4 = radar_features[2]

        image_feat5 = image_features[3]
        radar_feat5 = radar_features[3]
        # --------------------------------------------------------------------------- #
        # ============================================================================ #

        # ===================== the second asymmetric fair fusion ==================== #
        # --------------------- radar after enhancement 1 128*140*140 ---------------- #
        image_feat_prepared_to_radar2 = self.vision_to_radar_attn2(image_feat2)
        radar_feat2_detection = torch.cat([image_feat_prepared_to_radar2, radar_feat2], dim=1)
        # radar_feat2_detection = shuffle_channels(radar_feat2_detection, groups=2)
        # radar_feat2_detection = eca_block(radar_feat2_detection)
        radar_feat2_detection = self.inverse_projection1(radar_feat2_detection)
        # --------------- long residual -------------- #
        radar_feat2_detection = radar_feat2_detection + image_feat_prepared_to_radar2
        # -------------------------------------------- #

        # --------------------- image after enhancement 2 128*140*140 ---------------- #
        radar_feat_prepared_to_image2 = self.radar_to_vision_projection2(radar_feat2)
        image_feat2_segmentation = (1 + data_normal(radar_feat_prepared_to_image2)) * image_feat2
        # ---------------------------------------------------------------------------- #
        # ============================================================================ #

        # ===================== the third asymmetric fair fusion ==================== #
        # --------------------- radar after enhancement 1 320*70*70 ---------------- #
        image_feat_prepared_to_radar3 = self.vision_to_radar_attn3(image_feat3)
        radar_feat3_detection = torch.cat([image_feat_prepared_to_radar3, radar_feat3], dim=1)
        # radar_feat3_detection = shuffle_channels(radar_feat3_detection, groups=2)
        # radar_feat3_detection = eca_block(radar_feat3_detection)
        radar_feat3_detection = self.inverse_projection2(radar_feat3_detection)
        # --------------- long residual -------------- #
        radar_feat3_detection = radar_feat3_detection + image_feat_prepared_to_radar3
        # -------------------------------------------- #

        # --------------------- image after enhancement 2 320*70*70 ---------------- #
        radar_feat_prepared_to_image3 = self.radar_to_vision_projection3(radar_feat3)
        image_feat3_segmentation = (1 + data_normal(radar_feat_prepared_to_image3)) * image_feat3
        # ---------------------------------------------------------------------------- #
        # ============================================================================ #

        # ===================== the fourth asymmetric fair fusion ==================== #
        # --------------------- radar after enhancement 1 512*35*35 ---------------- #
        image_feat_prepared_to_radar4 = self.vision_to_radar_attn4(image_feat4)
        radar_feat4_detection = torch.cat([image_feat_prepared_to_radar4, radar_feat4], dim=1)
        # radar_feat4_detection = shuffle_channels(radar_feat4_detection, groups=2)
        # radar_feat4_detection = eca_block(radar_feat4_detection)
        radar_feat4_detection = self.inverse_projection3(radar_feat4_detection)
        # --------------- long residual -------------- #
        radar_feat4_detection = radar_feat4_detection + image_feat_prepared_to_radar4
        # -------------------------------------------- #

        # --------------------- image after enhancement 2 512*35*35 ---------------- #
        radar_feat_prepared_to_image4 = self.radar_to_vision_projection4(radar_feat4)
        image_feat4_segmentation = (1 + data_normal(radar_feat_prepared_to_image4)) * image_feat4
        # ---------------------------------------------------------------------------- #
        # ============================================================================ #

        # ===================== the fifth asymmetric fair fusion ==================== #
        # --------------------- radar after enhancement 1 512*17*17 ---------------- #
        image_feat_prepared_to_radar5 = self.vision_to_radar_attn5(image_feat5)
        radar_feat5_detection = torch.cat([image_feat_prepared_to_radar5, radar_feat5], dim=1)
        # radar_feat5_detection = shuffle_channels(radar_feat5_detection, groups=2)
        # radar_feat5_detection = eca_block(radar_feat5_detection)
        radar_feat5_detection = self.inverse_projection4(radar_feat5_detection)
        # --------------- long residual -------------- #
        radar_feat5_detection = radar_feat5_detection + image_feat_prepared_to_radar5
        # -------------------------------------------- #

        # --------------------- image after enhancement 2 512*35*35 ---------------- #
        radar_feat_prepared_to_image5 = self.radar_to_vision_projection5(radar_feat5)
        image_feat5_segmentation = (1 + data_normal(radar_feat_prepared_to_image5)) * image_feat5
        # ---------------------------------------------------------------------------- #
        # ============================================================================ #

        return radar_feat2_detection, radar_feat3_detection, radar_feat4_detection, radar_feat5_detection, \
               image_feat2_segmentation, image_feat3_segmentation, image_feat4_segmentation, image_feat5_segmentation


if __name__ == '__main__':
    input = torch.rand(1, 3, 640, 640)
    input_radar = torch.rand(1, 4, 640, 640)
    model = VRCoC()
    out = model(input, input_radar)
    # print(model)
    print(summary(model, input_size=[(1, 3, 640, 640), (1, 4, 640, 640)]))
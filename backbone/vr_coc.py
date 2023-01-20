from backbone.radar.context_cluster import coc_small, coc_medium, coc_tiny2
from backbone.vision.context_cluster import coc_small, coc_medium, coc_tiny2
from backbone.vision.context_cluster import ClusterBlock
from backbone.attention_modules.shuffle_attention import ShuffleAttention
from neck.fpnt_segmentation import BaseConv

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class VRCoC(nn.Module):
    def __init__(self, version='s', backbone_pretrain=False):
        super(VRCoC, self).__init__()

        self.vision_initial = ClusterBlock(dim=3)
        self.radar_initial = ClusterBlock(dim=4)
        self.vision_to_radar_attn1 = BaseConv(in_channels=3, out_channels=3, ksize=1, stride=1)
        self.radar_to_vision_projection = BaseConv(in_channels=4, out_channels=1, ksize=1, stride=1)

        if version == 'tiny':
            self.radar_backbone = coc_tiny2(pretrained=backbone_pretrain)
            self.vision_backbone = coc_tiny2(pretrained=backbone_pretrain)
            self.in_channels = [64, 196, 320]
        elif version == 's':
            self.radar_backbone = coc_small(pretrained=backbone_pretrain)
            self.vision_backbone = coc_small(pretrained=backbone_pretrain)
            self.in_channels = [128, 320, 512]
        elif version == 'm':
            self.radar_backbone = coc_medium(pretrained=backbone_pretrain)
            self.vision_backbone = coc_medium(pretrained=backbone_pretrain)
            self.in_channels = [128, 320, 512]














    def forward(self, x):
        return x
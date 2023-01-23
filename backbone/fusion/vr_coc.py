"""
ContextCluster implementation
"""
import os
import copy
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from einops import rearrange
import torch.nn.functional as F
from backbone.attention_modules.shuffle_attention import ShuffleAttention
from backbone.attention_modules.eca import eca_block
from backbone.conv_utils.normal_conv import BaseConv
from torchinfo import summary
from thop import profile
from thop import clever_format


try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'model_small': _cfg(crop_pct=0.9),
    'model_medium': _cfg(crop_pct=0.95),
}


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


class PointRecuder(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2:torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1,dim=-1)
    x2 = F.normalize(x2,dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False):
        """

        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim=head_dim
        self.fc1 = nn.Conv2d(dim, heads*head_dim, kernel_size=1)
        self.fc2 = nn.Conv2d(heads*head_dim, out_dim, kernel_size=1)
        self.fc_v = nn.Conv2d(dim, heads*head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w,proposal_h))
        self.fold_w=fold_w
        self.fold_h = fold_h
        self.return_center = return_center

    def forward(self, x): #[b,c,w,h]
        value = self.fc_v(x)
        x = self.fc1(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w>1 and self.fold_h>1:
            # split the big feature maps to small loca regions to reduce computations of matrix multiplications.
            b0,c0,w0,h0 = x.shape
            assert w0%self.fold_w==0 and h0%self.fold_h==0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h) #[bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b,c,w,h = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
        value_centers = rearrange(self.centers_proposal(value) , 'b c w h -> b (w h) c') # [b,C_W,C_H,c]
        b,c,ww,hh = centers.shape
        sim = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(centers.reshape(b,c,-1).permute(0,2,1), x.reshape(b,c,-1).permute(0,2,1))) #[B,M,N]
        # sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1,keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim= sim*mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
        # out shape [B,M,D]
        out = ( ( value2.unsqueeze(dim=1)*sim.unsqueeze(dim=-1) ).sum(dim=2) + value_centers)/ (mask.sum(dim=-1,keepdim=True)+ 1.0) # [B,M,D]

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        # return to each point in a cluster
        else:
            out = (out.unsqueeze(dim=2)*sim.unsqueeze(dim=-1)).sum(dim=1) # [B,N,D]
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w>1 and self.fold_h>1: # recover the splited regions back to big feature maps
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.fc2(out)
        return out


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False):

        super().__init__()

        self.norm1 = norm_layer(dim)
        # dim, out_dim, proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False
        self.token_mixer = Cluster(dim=dim, out_dim=dim, proposal_w=proposal_w,proposal_h=proposal_h,
                                   fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim, return_center=False)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep ContextClusters.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(ClusterBlock(
            dim, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
            heads=heads, head_dim=head_dim, return_center=False
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class ImageEnhanceByRadar(nn.Module):
    def __init__(self, radar_in_channels, image_in_channels):
        super(ImageEnhanceByRadar, self).__init__()
        self.radar_in_channels = radar_in_channels
        self.image_in_channels = image_in_channels
        self.radar_projection = BaseConv(in_channels=self.radar_in_channels, out_channels=1, ksize=1, stride=1)
        self.norm = nn.BatchNorm2d(self.image_in_channels)

    def forward(self, image_map, radar_map):
        key_points_normal = self.radar_projection(radar_map)
        key_image_map = (1 + data_normal(key_points_normal)) * image_map
        key_image_map = self.norm(key_image_map)
        return key_image_map


class RadarEnhanceByImage(nn.Module):
    def __init__(self, radar_in_channels, image_in_channels, initial=False):
        super(RadarEnhanceByImage, self).__init__()
        self.initial = initial
        self.radar_in_channels = radar_in_channels
        self.image_in_channels = image_in_channels
        self.image_attn = ShuffleAttention(channel=self.image_in_channels, G=2)
        self.channel_attn = eca_block(channel=self.radar_in_channels + self.image_in_channels)
        self.inverse_projection = BaseConv(in_channels=self.radar_in_channels + self.image_in_channels,
                                           out_channels=radar_in_channels, ksize=1, stride=1)
        self.norm = nn.BatchNorm2d(self.radar_in_channels)

    def forward(self, image_map, radar_map):
        if self.initial:
            # ------------------- concatenate maps and shuffle with channel attention ---------------------- #
            image_radar_maps = torch.cat([image_map, radar_map], axis=1)
            image_radar_maps = shuffle_channels(image_radar_maps, 2)
            image_radar_maps = self.channel_attn(image_radar_maps)
            # ---------------------------------------------------------------------------------------------- #
            # -------------------------- inverse projection with a long residual path ---------------------- #
            image_radar_maps = self.inverse_projection(image_radar_maps)
            image_radar_maps = image_radar_maps + radar_map
            image_radar_maps = self.norm(image_radar_maps)
            # ---------------------------------------------------------------------------------------------- #

        else:
            # --------------------- image map with both spatial and channel attention ---------------------- #
            image_map_attn = self.image_attn(image_map)
            # ---------------------------------------------------------------------------------------------- #
            # ------------------- concatenate maps and shuffle with channel attention ---------------------- #
            image_radar_maps = torch.cat([image_map_attn, radar_map], axis=1)
            image_radar_maps = shuffle_channels(image_radar_maps, 2)
            image_radar_maps = self.channel_attn(image_radar_maps)
            # ---------------------------------------------------------------------------------------------- #
            # -------------------------- inverse projection with a long residual path ---------------------- #
            image_radar_maps = self.inverse_projection(image_radar_maps)
            image_radar_maps = image_radar_maps + radar_map
            image_radar_maps = self.norm(image_radar_maps)
            # ---------------------------------------------------------------------------------------------- #

        return image_radar_maps


class VRCoC(nn.Module):
    """
    ContextCluster, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, the embedding dims, mlp ratios
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained:
        for mmdetection and mmsegmentation to load pretrained weights
    """
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 num_classes=1000,
                 in_patch_size=4, in_stride=4, in_pad=0,
                 down_patch_size=2, down_stride=2, down_pad=0,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=True,
                 init_cfg=None,
                 pretrained=None,
                 # the parameters for context-cluster
                 img_w=512,img_h=512,
                 proposal_w=[2,2,2,2], proposal_h=[2,2,2,2], fold_w=[8,4,2,1], fold_h=[8,4,2,1],
                 heads=[2,4,6,8], head_dim=[16,16,32,32],
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        # register positional information buffer of image.
        range_w = torch.arange(0, img_w, step=1)/(img_w-1.0)
        range_h = torch.arange(0, img_h, step=1)/(img_h-1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h), dim = -1).float()
        fea_pos = fea_pos-0.5
        self.register_buffer('fea_pos', fea_pos)

        # register positional information buffer of radar map.
        range_w_r = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h_r = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        fea_pos_r = torch.stack(torch.meshgrid(range_w_r, range_h_r), dim=-1).float()
        fea_pos_r = fea_pos_r - 0.5
        self.register_buffer('fea_pos_r', fea_pos_r)

        self.image_initial = PointRecuder(patch_size=1, stride=1, padding=0,
            in_chans=3, embed_dim=3)

        self.radar_initial = PointRecuder(patch_size=1, stride=1, padding=0,
            in_chans=4, embed_dim=4)

        self.radar_enhance_by_image1 = RadarEnhanceByImage(image_in_channels=3, radar_in_channels=4, initial=True)
        self.image_enhance_by_radar1 = ImageEnhanceByRadar(image_in_channels=3, radar_in_channels=4)

        self.patch_embed = PointRecuder(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=5, embed_dim=embed_dims[0])

        self.patch_embed_radar = PointRecuder(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=6, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        network_radar = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 proposal_w=proposal_w[i],proposal_h=proposal_h[i],
                                 fold_w=fold_w[i], fold_h=fold_h[i], heads=heads[i], head_dim=head_dim[i],
                                 return_center=False
                                 )
            network.append(stage)

            stage_radar = basic_blocks(embed_dims[i], i, layers,
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 proposal_w=proposal_w[i], proposal_h=proposal_h[i],
                                 fold_w=fold_w[i], fold_h=fold_h[i], heads=heads[i], head_dim=head_dim[i],
                                 return_center=False
                                 )
            network_radar.append(stage_radar)

            image_enhance = ImageEnhanceByRadar(image_in_channels=embed_dims[i], radar_in_channels=embed_dims[i])
            network.append(image_enhance)

            radar_enhance = RadarEnhanceByImage(image_in_channels=embed_dims[i], radar_in_channels=embed_dims[i])
            network_radar.append(radar_enhance)

            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PointRecuder(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

                network_radar.append(
                    PointRecuder(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)
        self.network_radar = nn.ModuleList(network_radar)
        # print("network length:", len(self.network))
        # print(list(self.network)[2])
        # print("network_radar length:", len(self.network_radar))
        # print(list(self.network_radar)[2])

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 3, 6, 9]
            # for i_emb, i_layer in enumerate(self.out_indices):
            #     if i_emb == 0 and os.environ.get('FORK_LAST3', None):
            #         # TODO: more elegant way
            #         """For RetinaNet, `start_level=1`. The first norm layer will not used.
            #         cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
            #         """
            #         layer = nn.Identity()
            #     else:
            #         layer = norm_layer(embed_dims[i_emb])
            #     layer_name = f'norm{i_layer}'
            #     self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x, x_radar):
        x = self.image_initial(x)
        x_radar = self.radar_initial(x_radar)

        x = self.image_enhance_by_radar1(x, x_radar)
        x_radar = self.radar_enhance_by_image1(x, x_radar)

        pos = self.fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)
        x = self.patch_embed(torch.cat([x, pos], dim=1))

        pos_radar = self.fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x_radar.shape[0], -1, -1, -1)
        x_radar = self.patch_embed_radar(torch.cat([x_radar, pos_radar], dim=1))
        return x, x_radar

    def forward_tokens(self, x, x_radar):
        outs = []
        outs_radar = []

        # ======================= the first stage (backbone, fusion, reducer) ======================== #
        # --------------------- backbone --------------------- #
        x = self.network[0](x)
        x_radar = self.network_radar[0](x_radar)
        # ---------------------------------------------------- #

        # --------------------- fusion ----------------------- #
        x = self.network[1](x, x_radar)
        x_radar = self.network_radar[1](x, x_radar)
        x_out = x
        x_radar_out = x_radar
        outs.append(x_out)
        outs_radar.append(x_radar_out)
        # ---------------------------------------------------- #

        # -------------------- reducer ----------------------- #
        x = self.network[2](x)
        x_radar = self.network_radar[2](x_radar)
        x_out1 = x
        x_radar_out1 = x_radar
        outs.append(x_out1)
        outs_radar.append(x_radar_out1)
        # ---------------------------------------------------- #
        # ============================================================================================ #

        # ======================= the second stage (backbone, fusion, reducer) ======================== #
        # --------------------- backbone --------------------- #
        x = self.network[3](x)
        x_radar = self.network_radar[3](x_radar)
        # ---------------------------------------------------- #

        # --------------------- fusion ----------------------- #
        x = self.network[4](x, x_radar)
        x_radar = self.network_radar[4](x, x_radar)
        # ---------------------------------------------------- #

        # -------------------- reducer ----------------------- #
        x = self.network[5](x)
        x_radar = self.network_radar[5](x_radar)
        x_out2 = x
        x_radar_out2 = x_radar
        outs.append(x_out2)
        outs_radar.append(x_radar_out2)
        # ---------------------------------------------------- #
        # ============================================================================================ #

        # ======================= the third stage (backbone, fusion, reducer) ======================== #
        # --------------------- backbone --------------------- #
        x = self.network[6](x)
        x_radar = self.network_radar[6](x_radar)
        # ---------------------------------------------------- #

        # --------------------- fusion ----------------------- #
        x = self.network[7](x, x_radar)
        x_radar = self.network_radar[7](x, x_radar)
        # ---------------------------------------------------- #

        # -------------------- reducer ----------------------- #
        x = self.network[8](x)
        x_radar = self.network_radar[8](x_radar)
        x_out3 = x
        x_radar_out3 = x_radar
        # outs.append(x_out3)
        # outs_radar.append(x_radar_out3)
        # ---------------------------------------------------- #
        # ============================================================================================ #

        # ======================= the fourth stage (backbone, fusion, reducer) ======================== #
        # --------------------- backbone --------------------- #
        x = self.network[9](x)
        x_radar = self.network_radar[9](x_radar)
        # ---------------------------------------------------- #

        # --------------------- fusion ----------------------- #
        x = self.network[10](x, x_radar)
        x_radar = self.network_radar[10](x, x_radar)
        x_out4 = x
        x_radar_out4 = x_radar
        outs.append(x_out4)
        outs_radar.append(x_radar_out4)
        # ---------------------------------------------------- #
        # ============================================================================================ #
        return outs, outs_radar


        # outs = []
        # for idx, block in enumerate(self.network):
        #     x = block(x)
        #     if self.fork_feat and idx in self.out_indices:
        #         norm_layer = getattr(self, f'norm{idx}')
        #         x_out = norm_layer(x)
        #         outs.append(x_out)
        # if self.fork_feat:
        #     # output the features of four stages for dense prediction
        #     return outs
        # # output only the features of last layer for image classification
        # return x

    def forward(self, x, x_radar):
        # input embedding
        x, x_radar = self.forward_embeddings(x, x_radar)
        # print(x.shape)
        # print(x_radar.shape)
        # through backbone
        x, x_radar = self.forward_tokens(x, x_radar)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x, x_radar
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out


@register_model
def coc_tiny(pretrained=False, **kwargs):
    layers = [3, 4, 5, 2]
    norm_layer=GroupNorm
    embed_dims = [32, 64, 196, 320]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w=[2,2,2,2]
    proposal_h=[2,2,2,2]
    fold_w=[8,4,2,1]
    fold_h=[8,4,2,1]
    heads=[4,4,8,8]
    head_dim=[24,24,24,24]
    down_patch_size=3
    down_pad = 1
    model = VRCoC(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size = down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_tiny2(pretrained=False, **kwargs):
    layers = [3, 4, 5, 2]
    norm_layer=GroupNorm
    embed_dims = [32, 64, 196, 320]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w=[4,2,7,4]
    proposal_h=[4,2,7,4]
    fold_w=[8,8,1,1]
    fold_h=[8,8,1,1]
    heads=[4,4,8,8]
    head_dim=[24,24,24,24]
    down_patch_size=3
    down_pad = 1
    model = VRCoC(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size = down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_small(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    norm_layer=GroupNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w=[2,2,2,2]
    proposal_h=[2,2,2,2]
    fold_w=[8,4,2,1]
    fold_h=[8,4,2,1]
    heads=[4,4,8,8]
    head_dim=[32,32,32,32]
    down_patch_size=3
    down_pad = 1
    model = VRCoC(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size = down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_medium(pretrained=False, width=1.0, **kwargs):
    layers = [4, 4, 12, 4]
    norm_layer=GroupNorm
    embed_dims = [int(64*width), int(128*width), int(320*width), int(512*width)]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w=[2,2,2,2]
    proposal_h=[2,2,2,2]
    fold_w=[8,4,2,1]
    fold_h=[8,4,2,1]
    heads=[6,6,12,12]
    head_dim=[32,32,32,32]
    down_patch_size=3
    down_pad = 1
    model = VRCoC(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size = down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


if __name__ == '__main__':
    # input = torch.rand(1, 64, 168, 168)
    # input_radar = torch.rand(1, 64, 168, 168)
    # model = RadarEnhanceByImage(channels=64)
    # output = model(input, input_radar)
    # print(output.shape)
    input = torch.rand(1, 3, 512, 512).cuda()
    input_radar = torch.rand(1, 4, 512, 512).cuda()
    model = coc_medium(width=0.25).cuda()
    out = model(input, input_radar)
    # print(model)
    print(len(out))
    print(out[0][0].shape)
    print(out[0][1].shape)
    print(out[0][2].shape)
    print(out[0][3].shape)
    print(summary(model, input_size=[(1, 3, 512, 512), (1, 4, 512, 512)]))

    macs, params = profile(model, inputs=(input, input_radar))
    macs, params = clever_format([macs, params], "%.3f")
    print("params:", params)
    print("macs:", macs)

    # input2 = torch.randn(1, 3, 672, 672)
    # coc_block = ClusterBlock(dim=3)
    # output2 = coc_block(input2)
    # print(summary(coc_block, input_size=(1, 3, 672, 672)))

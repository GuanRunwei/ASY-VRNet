import torch
import torch.nn as nn
import torch.nn.functional as F
from neck.coc_fpn_dual import CoCFpnDual
from head.decouplehead import DecoupleHead
from torchinfo import summary
from thop import profile
from thop import clever_format
# from torchsummary import summary
import time


class EfficientVRNet(nn.Module):
    def __init__(self, num_classes, num_seg_classes,  phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00}
        depth, width = depth_dict[phi], width_dict[phi]

        self.backbone = CoCFpnDual(width=width, num_seg_class=num_seg_classes)

        self.head = DecoupleHead(num_classes, width, depthwise=True)

    def forward(self, x, x_radar):
        fpn_outs, seg_outputs = self.backbone.forward(x, x_radar)
        det_outputs = self.head.forward(fpn_outs)
        return det_outputs, seg_outputs


if __name__ == '__main__':
    model = EfficientVRNet(num_classes=4, phi='l', num_seg_classes=5).cuda()
    model.eval()
    input_map1 = torch.randn((1, 3, 512, 512)).cuda()
    input_map2 = torch.randn((1, 4, 512, 512)).cuda()
    t1 = time.time()
    test_times = 300
    for i in range(test_times):
        output_map, output_seg = model(input_map1, input_map2)
    t2 = time.time()
    print("fps:", (1 / ((t2 - t1) / test_times)))

    output_map, output_seg = model(input_map1, input_map2)
    # print(output_map[0].shape)
    # print(output_map[1].shape)
    # print(output_map[2].shape)
    # print(output_seg.shape)
    print(summary(model, input_size=((1, 3, 512, 512), (1, 4, 512, 512))))

    macs, params = profile(model, inputs=([input_map1, input_map2]))
    macs, params = clever_format([macs, params], "%.3f")
    print("params:", params)
    print("macs:", macs)
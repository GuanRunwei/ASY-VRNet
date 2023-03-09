import torch
import torch.nn as nn
import math


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super().__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, loss_seg, loss_det):
        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss_det + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * loss_seg + self.log_vars[1]

        return loss0 + loss1


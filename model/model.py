"""
Author: HaoZhi
Date: 2024-03-08 15:49:22
LastEditors: HaoZhi
LastEditTime: 2024-03-08 15:58:28
Description: 
"""
from torch import nn

from timm.models.resnet import resnet34


class ResNet34(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.model = resnet34(num_classes=num_classes)

    def forward(self, x):
        logits = self.model(x)
        return logits

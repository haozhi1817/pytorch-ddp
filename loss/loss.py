"""
Author: HaoZhi
Date: 2024-03-08 16:00:18
LastEditors: HaoZhi
LastEditTime: 2024-03-08 16:01:12
Description: 
"""
from torch import nn


class CE_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.loss(logits, labels)

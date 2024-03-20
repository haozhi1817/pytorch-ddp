import os
import sys

sys.path.append(os.path.dirname(__file__))

from torch import nn

from model.model import ResNet34


class Model(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        if model_cfg["model"] == "res34":
            self.model = ResNet34(model_cfg["num_classes"])
        else:
            raise NotImplementedError

    def forward(self, inputs):
        logits = self.model(inputs)
        return logits

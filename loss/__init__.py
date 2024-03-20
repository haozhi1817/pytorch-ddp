import os
import sys

sys.path.append(os.path.dirname(__file__))

from torch import nn

from loss.loss import CE_Loss


class Loss(nn.Module):
    def __init__(self, loss_cfg) -> None:
        super().__init__()
        if loss_cfg["ce_loss_weight"] != 0:
            print("Build CE_Loss")
            self.loss_fn = CE_Loss()
            self.loss_weight = loss_cfg["ce_loss_weight"]
        else:
            raise NotImplementedError

    def forward(self, logits, labels):
        loss = self.loss_fn(logits, labels)
        total_loss = self.loss_weight * loss
        return dict(total_loss=total_loss, loss=loss)

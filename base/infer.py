"""
Author: HaoZhi
Date: 2024-03-08 16:35:49
LastEditors: HaoZhi
LastEditTime: 2024-03-08 16:36:54
Description: 
"""
import yaml

import torch

from engine import Base


def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    enginer = Base(cfg=cfg, if_training=False)
    valid_loader = enginer.valid_loader

    enginer.load_ckpt()

    n_epochs = enginer.train_cfg["num_epochs"]

    with torch.no_grad():
        enginer.model.eval()
        for batch_id, (batch_imgs, batch_labels) in enumerate(valid_loader):
            batch_probs, batch_preds = enginer.valid_one_step(batch_imgs)
            enginer.metric_one_step(batch_preds, batch_labels)
        acc = enginer.metric_total_data()
        print("ACC: ", acc)


if __name__ == "__main__":
    cfg_path = "/disk2/haozhi/tmp/code/dist_train/config.yaml"
    main(cfg_path)

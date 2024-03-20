"""
Author: HaoZhi
Date: 2024-03-08 16:30:32
LastEditors: HaoZhi
LastEditTime: 2024-03-08 16:34:46
Description: 
"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from data.dataset import CustomDataSet
from model import Model
from loss import Loss


class Base:
    def __init__(self, cfg, if_training) -> None:
        self.cfg = cfg
        self.data_cfg = self.cfg["data"]
        self.model_cfg = self.cfg["model"]
        self.loss_cfg = self.cfg["loss"]
        self.train_cfg = self.cfg["train"]
        self.valid_cfg = self.cfg["valid"]
        self.if_training = if_training
        self.time_stamp = "_".join(time.ctime().split(" "))
        if self.if_training:
            self.device = self.train_cfg["device"]
        else:
            self.device = self.valid_cfg["device"]
        self.valid_info = []

        if self.if_training:
            self.build_dataloader()
            self.build_model()
            self.build_loss()
            self.build_optimizer()
            self.build_lr_scheduler()
            self.build_summary()
            self.build_amp()
        else:
            self.build_dataloader()
            self.build_model()

    def build_dataloader(
        self,
    ):
        if self.if_training:
            train_set = CustomDataSet(self.data_cfg["train_folder"], True)
            self.train_loader = DataLoader(
                train_set,
                batch_size=self.train_cfg["batch_size"],
                drop_last=False,
                shuffle=True,
            )
        valid_set = CustomDataSet(self.data_cfg["valid_folder"], False)
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=self.valid_cfg["batch_size"],
            drop_last=False,
            shuffle=False,
        )

    def build_model(
        self,
    ):
        self.model = Model(self.model_cfg)
        self.model.to(self.device)

    def build_loss(
        self,
    ):
        self.loss_f = Loss(self.loss_cfg)

    def build_optimizer(
        self,
    ):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg["lr"],
            weight_decay=self.train_cfg["wd"],
        )

    def build_lr_scheduler(
        self,
    ):
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=self.train_cfg["step_lr_scheduler"]["step"],
            gamma=self.train_cfg["step_lr_scheduler"]["gamma"],
        )

    def build_amp(
        self,
    ):
        if self.train_cfg["amp"]:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def build_summary(
        self,
    ):
        log_dir = os.path.join(
            self.train_cfg["log_dir"], self.train_cfg["version"], self.time_stamp
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.summary_wariter = SummaryWriter(log_dir=log_dir)

    def update_summary(self, step):
        self.summary_wariter.add_scalar(
            "lr", self.lr_scheduler.get_last_lr()[0], global_step=step
        )
        for k, v in self.loss_dict.items():
            self.summary_wariter.add_scalar(k, v.item(), global_step=step)

    def load_ckpt(
        self,
    ):
        if self.if_training:
            resume_path = self.train_cfg["resume"]
        else:
            resume_path = self.valid_cfg["resume"]
        if os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location=self.device)
            resume_state = ckpt["model"]
            model_state = self.model.state_dict()
            for k, v in model_state.items():
                if k in resume_state and resume_state[k].shape == v.shape:
                    model_state[k] = resume_state[k]
                else:
                    print(k, ": ", v.shape, "not match resume: ", resume_state[k].shape)
            self.model.load_state_dict(model_state)

    def save_ckpt(self, epoch):
        self.ckpt_dir = os.path.join(
            self.train_cfg["ckpt_dir"], self.train_cfg["version"], self.time_stamp
        )
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        ckpt = {"model": self.model.state_dict(), "epoch": epoch}
        torch.save(ckpt, os.path.join(self.ckpt_dir, "model_" + str(epoch) + ".pth"))

    def train_one_step(self, batch_imgs, batch_labels, current_step):
        batch_imgs = batch_imgs.float().to(self.device)
        batch_labels = batch_labels.long().to(self.device)
        self.optimizer.zero_grad()
        if self.scaler:
            with torch.cuda.amp.autocast():
                batch_logits = self.model(batch_imgs)
                self.loss_dict = self.loss_f(batch_logits, batch_labels)
            loss = self.loss_dict["total_loss"]
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            batch_logits = self.model(batch_imgs)
            self.loss_dict = self.loss_f(batch_logits, batch_labels)
            loss = self.loss_dict["total_loss"]
            loss.backward()
            self.optimizer.step()
        self.lr_scheduler.step()
        self.update_summary(current_step)
        print(
            "|Steps: %05d | Lr: %.8f |Total_Loss: %.5f |"
            % (
                current_step,
                self.lr_scheduler.get_last_lr()[0],
                self.loss_dict["total_loss"].item(),
            )
        )

    def valid_one_step(self, batch_imgs):
        batch_imgs = batch_imgs.to(self.device)
        batch_logits = self.model(batch_imgs)
        batch_probs = batch_logits.softmax(1)
        batch_preds = batch_logits.argmax(1)
        return batch_probs, batch_preds

    def metric_one_step(self, batch_preds, batch_labels):
        batch_accs = (batch_preds == batch_labels.to(self.device)).float()
        self.valid_info.append(batch_accs)

    def metric_total_data(
        self,
    ):
        acc = torch.concat(self.valid_info, dim=0)
        return acc.mean().cpu().numpy()

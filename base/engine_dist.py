import os
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from data.dataset import CustomDataSet
from model import Model
from loss import Loss


class Base:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.data_cfg = self.cfg["data"]
        self.train_cfg = self.cfg["train"]
        self.model_cfg = self.cfg["model"]
        self.loss_cfg = self.cfg["loss"]
        self.time_stamp = "_".join(time.ctime().split(" "))

        self.__init_dist_train()
        self.__build_dataloader()
        self.__build_model()
        self.__load_ckpt()
        self.__build_optimizer()
        self.__build_lr_scheduler()
        self.__build_ddp_model()
        self.__build_loss()
        self.__build_amp()
        self.__build_summary()

    def __init_dist_train(self):
        print("Init Dist Process")
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size
        )
        dist.barrier()

    def __build_dataloader(
        self,
    ):
        print("Build Train DataLoader")
        train_dataset = CustomDataSet(
            data_foldr=self.data_cfg["train_folder"], if_aug=True
        )
        train_sampler = DistributedSampler(
            dataset=train_dataset, shuffle=True, drop_last=False
        )
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            sampler=train_sampler,
            batch_size=self.train_cfg["batch_size"],
        )
        self.num_train_data = len(train_dataset)

        print("Build Valid DataLoader")
        valid_dataset = CustomDataSet(
            data_foldr=self.data_cfg["valid_folder"], if_aug=False
        )
        valid_sampler = DistributedSampler(
            dataset=valid_dataset, shuffle=False, drop_last=False
        )
        self.valid_dataloader = DataLoader(
            dataset=valid_dataset,
            sampler=valid_sampler,
            batch_size=self.train_cfg["batch_size"],
        )
        self.num_valid_data = len(valid_dataset)

    def __build_model(self):
        print("Build Model")
        model = Model(self.model_cfg).to(torch.device(self.local_rank))
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    def __build_ddp_model(
        self,
    ):
        self.ddp_model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.local_rank]
        )

    def __build_loss(self):
        print("Build Loss")
        self.loss = Loss(self.loss_cfg).to(torch.device(self.local_rank))

    def __build_optimizer(
        self,
    ):
        print("Build Optimizer")
        if self.train_cfg["opt"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.train_cfg["lr"],
                weight_decay=self.train_cfg["wd"],
            )
        elif self.train_cfg["opt"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.train_cfg["lr"],
                weight_decay=self.train_cfg["wd"],
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.train_cfg["lr"],
                weight_decay=self.train_cfg["wd"],
            )

    def __build_lr_scheduler(
        self,
    ):
        print("Build LR-Scheduler")
        if self.train_cfg["lr_scheduler"] == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.train_cfg["step_lr_scheduler"]["step"],
                gamma=self.train_cfg["step_lr_scheduler"]["gamma"],
            )
        elif self.train_cfg["lr_scheduler"] == "exp":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer,
                gamma=self.train_cfg["exp_lr_scheduler"]["gamma"],
            )
        else:
            raise NotImplementedError

    def __build_amp(
        self,
    ):
        if self.train_cfg["amp"]:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def __load_ckpt(
        self,
    ):
        """
        __load_ckpt 据说有三种load方式，一种是在ddp之前直接load(错误)，还有一种是利用map location分别load，还有一种是直接load到ddp.module上，不过不知道这种方式和第一种方式是否一致。
        """
        if os.path.isfile(self.train_cfg["resume"]):
            state_dict = torch.load(
                self.train_cfg["resume"], map_location=f"cuda:{self.local_rank}"
            )["model"]
            self.model.load_state_dict(state_dict)

    def __save_ckpt(self, epoch):
        save_folder = os.path.join(
            self.train_cfg["ckpt_dir"], self.train_cfg["version"], self.time_stamp
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if self.local_rank == 0:
            # state_dict = {'model': self.model.state_dict()}
            state_dict = {"model": self.ddp_model.module.state_dict()}
            torch.save(
                state_dict, os.path.join(save_folder, "model_" + str(epoch) + ".pth")
            )
        dist.barrier()

    def __build_summary(
        self,
    ):
        log_dir = os.path.join(
            self.cfg["train"]["log_dir"], self.cfg["train"]["version"], self.time_stamp
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=log_dir)

    def __update_summary(self, step):
        self.summary_writer.add_scalar(
            "lr", self.lr_scheduler.get_last_lr()[0], global_step=step
        )
        self.summary_info = {}
        for k, v in self.loss_dict.items():
            dist.reduce(tensor=v, dst=0, op=dist.ReduceOp.SUM)
            dist.barrier()
            if self.local_rank == 0:
                self.summary_writer.add_scalar(k, v.item(), global_step=step)
                self.summary_info[k] = v.item()

    def __acc(self, logits, labels):
        return (logits.argmax(1) == labels).float().sum()

    def train_one_epoch(self, epoch):
        self.ddp_model.train()
        self.train_dataloader.sampler.set_epoch(epoch)
        for batch_id, (imgs, masks) in enumerate(self.train_dataloader):
            current_step = epoch * len(self.train_dataloader) + batch_id
            imgs = imgs.to(torch.device(self.local_rank))
            masks = masks.to(torch.device(self.local_rank))
            self.optimizer.zero_grad()
            if self.scaler:
                with torch.cuda.amp.autocast():
                    batch_logits = self.ddp_model(imgs)
                    self.loss_dict = self.loss(batch_logits, masks)
                loss = self.loss_dict["total_loss"] / self.world_size
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.lr_scheduler.step()
            self.__update_summary(current_step)
            if self.local_rank == 0:
                print(
                    "|Steps: %05d | Lr: %.8f |Total_Loss: %.5f |Bin_Loss: %.5f |"
                    % (
                        current_step,
                        self.lr_scheduler.get_last_lr()[0],
                        self.loss_dict["total_loss"],
                        self.loss_dict["loss"],
                    )
                )
        self.__save_ckpt(epoch)

    def valid_one_epoch(self, epoch):
        total_acc = 0
        self.ddp_model.eval()
        with torch.inference_mode():
            for batch_id, (imgs, masks) in enumerate(self.valid_dataloader):
                imgs = imgs.to(torch.device(self.local_rank))
                masks = masks.to(torch.device(self.local_rank))
                logits = self.ddp_model(imgs)
                acc = self.__acc(logits, masks)
                dist.reduce(acc, dst=0)
                dist.barrier()
                total_acc += acc
        mean_acc = total_acc / self.num_valid_data
        if self.local_rank == 0:
            print("|Epochs: %05d | Acc: %.8f |" % (epoch, mean_acc))

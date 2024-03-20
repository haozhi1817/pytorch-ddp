"""
Author: HaoZhi
Date: 2024-03-08 14:52:27
LastEditors: HaoZhi
LastEditTime: 2024-03-08 15:08:54
Description: 
"""
import os
import glob

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CustomDataSet(Dataset):
    def __init__(self, data_foldr, if_aug) -> None:
        super().__init__()
        self.data_files = glob.glob(os.path.join(data_foldr, "*.jpg"))
        if if_aug:
            self.aug = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3), size=(224, 224)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.TrivialAugmentWide(
                        interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.ConvertImageDtype(torch.float),
                ]
            )
        else:
            self.aug = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ConvertImageDtype(torch.float),
                ]
            )

    def __getitem__(self, index):
        data_file = self.data_files[index]
        img = torch.from_numpy(np.array(Image.open(data_file).convert("RGB"))).permute(
            2, 0, 1
        )
        # print("before_aug: ", img.min(), img.max())
        img = self.aug(img)
        # print("after_aug: ", img.min(), img.max())
        label = int(data_file.split(os.sep)[-1].split("_")[1]) - 1
        return img, label

    def __len__(
        self,
    ):
        return len(self.data_files)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    dataset = CustomDataSet("/disk2/haozhi/tmp/data/102flowers/train", True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False)
    for idx, (img, label) in enumerate(dataloader):
        print(img.shape, img.max(), label)
        img = img.numpy().astype("float32").transpose(0, 2, 3, 1)
        for i in range(16):
            plt.imsave(
                os.path.join(
                    "/disk2/haozhi/tmp/code/dist_train/debug",
                    str(label[i].item()) + "_" + str(i) + ".jpg",
                ),
                img[i],
            )
        break

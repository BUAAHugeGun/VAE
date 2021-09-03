import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets
from torch.nn import functional as F
from torch import nn
from PIL import Image, ImageFilter
from PIL import ImageFile
from torchvision import transforms
import os
import numpy as np
import dataset.cifar10 as cifar10
import random
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import math
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.datasets import CelebA


class cifar10_dataset(Dataset):
    def __init__(self, path, train=True, **kwargs):
        super(cifar10_dataset, self).__init__()
        data_all = cifar10.load_CIFAR10(path)
        if train:
            self.set = [data_all[0], data_all[1]]
        else:
            self.set = [data_all[2], data_all[3]]
        labels = kwargs.get('labels')
        data_sum = kwargs.get('data_sum')
        if labels is not None:
            self.set[0] = [self.set[0][i] for i in range(len(self.set[0])) if self.set[1][i] in labels]
            self.set[1] = [self.set[1][i] for i in range(len(self.set[1])) if self.set[1][i] in labels]
        if data_sum is not None:
            self.set[0] = self.set[0][0:data_sum]
            self.set[1] = self.set[1][0:data_sum]
        self.transform = transforms.Compose(
            [transforms.Resize(64),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

    def __len__(self):
        return len(self.set[0])

    def __getitem__(self, id):
        return [self.transform(Image.fromarray(self.set[0][id].astype(np.uint8))), self.set[1][id]]


def build_data(tag, path, batch_size, training, num_worker, **kwargs):
    if tag == "mnist":
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
        ])
        mnist = tv_datasets.MNIST(root=path, train=training, transform=transform, download=False)
        dataloader = DataLoader(mnist, batch_size, shuffle=True, num_workers=num_worker)
        return dataloader
    elif tag == "cifar10":
        return DataLoader(cifar10_dataset(os.path.join(path, "cifar10"), **kwargs), batch_size, shuffle=True,
                          num_workers=num_worker)
    elif tag == "celeba":
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(kwargs['image_size']),
                                        transforms.ToTensor(),
                                        SetRange])
        print(path)
        return DataLoader(
            CelebA(root=os.path.join(path, 'celeba'), split="train" if training else "test", transform=transform,
                   download=True))
    else:
        assert 0


if __name__ == "__main__":
    data = build_data("celeba", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../data/"), 16, True,
                      1, image_size=64)
    print(next(iter(data)))

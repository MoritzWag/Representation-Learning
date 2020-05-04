import numpy as np 
import pandas as pd 
import argparse

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils import data
import torch
import torchvision
import os 
import pdb 


transforms = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(
                torchvision.datasets.MNIST(train=True,
                                            download=False,
                                            transform=transforms
                                            )
)
pdb.set_trace()
train_set, val_set = torch.utils.data.random_split(train_loader.dataset, (50000, 10000))


class MNISTData(data.Subset):

    def __init__(self, path, train, img_transform=None, attr_transform=None):
        self.path = path
        self.train = train
        self.img_transform = img_transform
        self.attr_transform = attr_transform

        if train:
            self.images = np.load(f"{path}/X_train.npy")
            self.attributes = np.load(f"{path}/Y_train.npy")
        else:
            self.images = np.load(f"{path}/X_test.npy")
            self.attributes = np.load(f"{path}/Y_test.npy")


    def __len__(self):
        return len(self.images.shape[0])

    def __getitem__(self, idx):
        img = self.images[idx, :, :, :]
        if self.img_transform:
            img = self.img_transform(img)
        attr = self.attr[idx]
        if self.attr_transform:
            attr = self.attr_transform(attr)
        
        return img, attr
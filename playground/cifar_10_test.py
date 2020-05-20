
import numpy as np 
import pandas as pd 
from PIL import Image
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable 
from library import utils
import torchvision.utils as vutils
import os

def run_checks():

    path = '/home/ubuntu/data/cifar10/'

    #transformation = transforms.Compose([transforms.ToPILImage(),
    #                                    transforms.CenterCrop(28),
    #                                    transforms.ToTensor()])

    transformation = transforms.Compose([transforms.ToTensor(),
                                        transforms.ToPILImage(),
                                        transforms.ToTensor()])

    train_rawdata, val_rawdata = utils.img_to_npy(path=path,
                                                train=True,
                                                val_split_ratio=0.2)

    train_data = utils.ImageData(rawdata=train_rawdata, transform=None)

    train_gen = DataLoader(dataset=train_data, 
                            batch_size=32, 
                            shuffle=False)
    
    storage_path = '../image/cifar10_test/'
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    
    for batch, (X, Y) in enumerate(train_gen):

        if batch % 250 == 0:
            vutils.save_image(X.data,
                            f"{storage_path}cifar10_{batch}.png",
                            normalize=True,
                            nrow=12)


if __name__ == "__main__":
    run_checks()
            

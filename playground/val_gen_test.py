import argparse
import numpy as np
import os
import pdb 

from library import utils
#from library.architectures import ConvEncoder, ConvDecoder, ConvEncoder28x28, ConvDecoder28x28
from torch.utils import data
from torch import optim 
from torchvision import transforms

path = '../data/mnist/'

train_rawdata, val_rawdata = utils.img_to_npy(path=path,
                                            train=True,
                                            val_split_ratio=0.2
                                            )

test_rawdata = utils.img_to_npy(path=path, train=False)



train_data = utils.ImageData(rawdata=train_rawdata)
val_data = utils.ImageData(rawdata=val_rawdata)
test_data = utils.ImageData(rawdata=test_rawdata)

train_gen = data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
val_gen = data.DataLoader(dataset=val_data, batch_size=32, shuffle=False)
test_gen = data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)


for batch, (X, Y) in enumerate(val_gen):
    if X.shape != (32, 1, 28, 28):
        print("ups!")

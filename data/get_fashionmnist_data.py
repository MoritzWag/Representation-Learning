import numpy as np 
import pandas as pd 
import argparse

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torchvision
import os 
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='fashionmnist')
    parser.add_argument('--path', type=str, default='.',
                        help='path to store data')
    args = parser.parse_args()
    return args

def get_data(args):
    data_path = os.path.expanduser(args.path)
    storage_path = '{}/fashionmnist/'.format(data_path)
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)
    
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=transform, target_transform=None, download=True)
    testset = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=transform, target_transform=None, download=True)

    X_train = trainset.data.unsqueeze(1) / 255.
    Y_train = pd.DataFrame(trainset.targets, columns=['labels'])
    
    X_test = testset.data.unsqueeze(1)/ 255.
    Y_test = pd.DataFrame(testset.targets, columns=['labels'])

    # store the data as .npy ndarray
    np.save(file='{}X_train.npy'.format(storage_path), arr=X_train)
    np.save(file='{}X_test.npy'.format(storage_path), arr=X_test)

    # store labels as .csv
    Y_train.to_csv('{}Y_train.csv'.format(storage_path))
    Y_test.to_csv('{}Y_test.csv'.format(storage_path))


    # something here is not working: command 'rm' is unrecognized!
    os.system('rmdir {}/processed; rmdir -rf {}/raw'.format(data_path, data_path))
    print('Stored fashionmnist data')

if __name__ == "__main__":
    args = parse_args()
    get_data(args=args)
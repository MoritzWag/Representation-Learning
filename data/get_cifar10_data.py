"""Download cifar10 data and bring it in our data format
"""
import numpy as np
import argparse
import pandas as pd
import torchvision.transforms as transforms
import torch
import torchvision
import os
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--path', type=str, default='/home/ubuntu/data',
                        help='path to store data')
    args = parser.parse_args()
    return args


def get_data(args):
    # normalize images
    data_path = os.path.expanduser(args.path)
    storage_path = '{}/cifar10/'.format(data_path)
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    # to npy and csv
    X_train = trainset.data.swapaxes(2, 3).swapaxes(1, 2)
    Y_train = trainset.targets
    X_test = testset.data.swapaxes(2, 3).swapaxes(1, 2)
    Y_test = testset.targets

    Y_train = pd.DataFrame(data=Y_train, columns=['labels'])
    Y_test = pd.DataFrame(data=Y_test, columns=['labels'])
    # store the data as .npy ndarray
    np.save(file='{}X_train.npy'.format(storage_path), arr=X_train)
    np.save(file='{}X_test.npy'.format(storage_path), arr=X_test)
    #np.save(file='{}Y_train.npy'.format(storage_path), arr=Y_train)
    #np.save(file='{}Y_test.npy'.format(storage_path), arr=Y_test)
    Y_train.to_csv('{}Y_train.csv'.format(storage_path))
    Y_test.to_csv('{}Y_test.csv'.format(storage_path))

    os.system('rm {}/cifar-10-python.tar.gz; rm -rf {}/cifar-10-batches-py'.format(data_path, data_path))
    print('Stored cifar10 data')


if __name__ == "__main__":
    args = parse_args()
    get_data(args=args)
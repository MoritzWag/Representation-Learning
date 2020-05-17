

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import argparse
import zipfile

import os
import random
import numpy as np
from PIL import Image
from random import shuffle
from scipy.misc import imresize



import torch
from torchvision.datasets.utils import download_file_from_google_drive
import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset
from torchvision import transforms



import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='celeba')
    parser.add_argument('--path', type=str, default='.',
                        help='path to store data')
    args = parser.parse_args()
    return args

def get_data(args):

    data_path = os.path.expanduser(args.path)
    storage_path = '{}/celeba/'.format(data_path)
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)

        download_file_from_google_drive("0B7EVK8r0v71pY0NSMzRuSXJEVkk",
                                        root=storage_path, filename='list_eval_partition.txt')
        
        download_file_from_google_drive("0B7EVK8r0v71pblRyaVFSWGxPY0U", 
                                        root=storage_path, filename='list_attr_celeba.txt')
    
        download_file_from_google_drive("0B7EVK8r0v71pZjFTYXZWM3FlRnM",
                                        root=storage_path, filename='img_align_celeba.zip')
        
        #download_file_from_google_drive("0B7EVK8r0v71peklHb0pGdDl6R28",
        #                                root=storage_path, filename='img_align_celeba.zip')


        ###############################





        ############################### 

        img_path = 'celeba/img/'
        if not os.path.exists(img_path):
            os.mkdir(img_path)
            
        with zipfile.ZipFile('celeba/img_align_celeba.zip', "r") as f:
            f.extractall(img_path)


    VALID_PARTITIONS = {'train': 0, 'val': 1, 'test': 2}
    ATTR_TO_IX_DICT = {'Sideburns': 30, 'Black_Hair': 8, 'Wavy_Hair': 33, 'Young': 39, 'Heavy_Makeup': 18, 
                    'Blond_Hair': 9, 'Attractive': 2, '5_o_Clock_Shadow': 0, 'Wearing_Necktie': 38, 
                    'Blurry': 10, 'Double_Chin': 14, 'Brown_Hair': 11, 'Mouth_Slightly_Open': 21, 
                    'Goatee': 16, 'Bald': 4, 'Pointy_Nose': 27, 'Gray_Hair': 17, 'Pale_Skin': 26, 
                    'Arched_Eyebrows': 1, 'Wearing_Hat': 35, 'Receding_Hairline': 28, 'Straight_Hair': 32, 
                    'Big_Nose': 7, 'Rosy_Cheeks': 29, 'Oval_Face': 25, 'Bangs': 5, 'Male': 20, 'Mustache': 22, 
                    'High_Cheekbones': 19, 'No_Beard': 24, 'Eyeglasses': 15, 'Bags_Under_Eyes': 3, 
                    'Wearing_Necklace': 37, 'Wearing_Lipstick': 36, 'Big_Lips': 6, 'Narrow_Eyes': 23, 
                    'Chubby': 13, 'Smiling': 31, 'Bushy_Eyebrows': 12, 'Wearing_Earrings': 34}
    ATTR_IX_TO_KEEP = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
    IX_TO_ATTR_DICT = {v:k for k,v in ATTR_TO_IX_DICT.items()}
    N_ATTRS = len(ATTR_IX_TO_KEEP)  

    # convert images to numpy array:
    # open PIL Image
    # image = Image.open(image_path)
    # image = image.convert('RGB')
    def load_eval_partition(partition):
        eval_data = []
        with open('celeba/list_eval_partition.txt') as fp:
            rows = fp.readlines()
            for row in rows:
                path, label = row.strip().split(' ')
                label = int(label)
                if label == VALID_PARTITIONS[partition]:
                    eval_data.append(path)
        return eval_data




    image_paths_train = load_eval_partition(partition='train')
    image_paths_val = load_eval_partition(partition='val')  
    image_paths_test = load_eval_partition(partition='test')


    img_transforms = transforms.Compose([transforms.Scale(64),
                                        transforms.CenterCrop(64),
                                        transforms.ToTensor()])

      
    X_train = []
    for index in range(len(image_paths_train)):
        if index % 100:
            print(index / len(image_paths_train))
        img_path = os.path.join('celeba/img/img_align_celeba/', image_paths_train[index])
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img_transforms(img).unsqueeze(0).numpy()
        X_train.append(img)
    
    X_train = np.stack(X_train)
    X_train = np.squeeze(X_train)


    X_val = []
    for index in range(len(image_paths_val)):
        img_path = os.path.join('celeba/img/img_align_celeba/', image_paths_val[index])
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img_transforms(img).unsqueeze(0).numpy()
        X_val.append(img)
    
    X_val = np.stack(X_val)
    X_val = np.squeeze(X_val)


    X_test = []
    for index in range(len(image_paths_val)):
        img_path = os.path.join('celeba/img/img_align_celeba', image_paths_test[index])
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img_transforms(img).unsqueeze(0).numpy()
        X_test.append(img)

    X_test = np.stack(X_test)
    X_test = np.squeeze(X_test)


    np.save(file='{}X_train.npy'.format(storage_path), arr=X_train)
    np.save(file='{}X_val.npy'.format(storage_path), arr=X_val)
    np.save(file='{}X_test.npy'.format(storage_path), arr=X_test)



    def load_attributes(paths, partition):
        if os.path.isfile('./data/Anno/attr_%s.npy' % partition):
            attr_data = np.load('./data/Anno/attr_%s.npy' % partition)
        else:
            attr_data = []
            with open('./data/Anno/list_attr_celeba.txt') as fp:
                rows = fp.readlines()
                for ix, row in enumerate(rows[2:]):
                    row = row.strip().split()
                path, attrs = row[0], row[1:]
                if path in paths:
                    attrs = np.array(attrs).astype(int)
                    attrs[attrs < 0] = 0
                attr_data.append(attrs)
        attr_data = np.vstack(attr_data).astype(np.int64)
        attr_data = torch.from_numpy(attr_data).float()
        return attr_data[:, ATTR_IX_TO_KEEP]


    def tensor_to_attributes(tensor):
        """
        @param tensor: PyTorch Tensor
                   D dimensional tensor
        @return attributes: list of strings
        """
        attrs = []
        n = tensor.size(0)
        tensor = torch.round(tensor)
        for i in xrange(n):
            if tensor[i] > 0.5:
                attr = IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[i]]
                attrs.append(attr)
        return attrs









if __name__ == "__main__":
    args = parse_args()
    get_data(args=args)


#class CelebAttributes(Dataset):
#    """Load images of celebrities and attributes."""
#    def __init__(self, partition='train', image_transform=None, attr_transform=None):
#        self.partition = partition
#        self.image_transform = image_transform
#        self.attr_transform = attr_transform
#        
#        assert partition in VALID_PARTITIONS.keys()
#        self.image_paths = load_eval_partition(partition)
#        self.attr_data = load_attributes(self.image_paths, partition)
#       self.size = int(len(self.image_paths))
#
#
#    def __getitem__(self, index):
#        """
#        Args:
#            index (int): Index
#        Returns:
#            tuple: (image, target) where target is index of the target class.
#        """
#        image_path = os.path.join('./data/img_align_celeba/', self.image_paths[index])
#        attr = self.attr_data[index]
#
#        # open PIL Image
#        image = Image.open(image_path)
#        image = image.convert('RGB')
#
#        if self.image_transform is not None:
#            image = self.image_transform(image)
#
#        if self.attr_transform is not None:
#            attr = self.attr_transform(attr)
#
#        return image, attr
#
#    def __len__(self):
#        return self.size









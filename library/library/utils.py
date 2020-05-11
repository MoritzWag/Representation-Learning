
from torch.utils import data

import numpy as np 
import pandas as pd 
import sklearn as sk 
import os 
import pdb 


######################
#
# Dataloader Functionalities
#
######################


class ImageData(data.Dataset):
	"""Image data in Pytorch is always decoded as (N,C,H,W)
	Args: 
		rawdata: {ndarray tuple} as provided by different data loader
	Returns:
		initiates the dataset object with instance attributes needed later
	"""

	def __init__(self, rawdata, transform=None):
		'Characterizes a dataset for PyTorch'
		self.rawdata = rawdata
		self.nclasses = len(np.unique(self.rawdata[1]))
		self.height = self.rawdata[0].shape[1]
		self.width = self.rawdata[0].shape[2]
		self.freq_classes = np.unique(self.rawdata[1], return_counts=True)
		self.transform =  transform

	def __len__(self):
		'Denotes the total number of samples'
		return(len(self.rawdata[0]))


	def __getitem__(self, idx):
		'Generates one sample of data'
		x = self.rawdata[0][idx, :, :, :]
		y = self.rawdata[1][idx]

		if self.transform:
			x = self.transform(x)
		
		return x, y




def img_to_npy(path, train=True, val_split_ratio=0.0):
	"""
	Args:
		path: {string} path to the dataset
		train: {bool} read train or test data
	Returns:
		(X, Y) ndarrays with dim(X) = (n_samples, n_channels(RGB), width, height)
		and dim(Y) = (n_samples)
	"""

	suffix = 'train' if train else 'test'

	X = np.load(file='{}X_{}.npy'.format(path, suffix)).astype('float64')
	Y = pd.read_csv('{}Y_{}.csv'.format(path, suffix))['labels'].values 

	classes = np.unique(Y)
	for idx in range(len(classes)):
		np.place(Y, Y == classes[idx], idx)
	Y = Y.astype(int)
	if val_split_ratio > 0.0:
		# split in val and train data
		idx_train = np.random.choice(a=X.shape[0],
			size=int((1-val_split_ratio) * X.shape[0]),
			replace=False)
		idx_val = [a for a in range(X.shape[0]) if a not in idx_train]

		X_val = X[idx_val, :, :, :]
		Y_val = Y[idx_val]

		X = X[idx_train, :, :, :]
		Y = Y[idx_train]


		return((X, Y), (X_val, Y_val))


	return(X, Y)




#########################
#
# Data Augmentation
#
#########################

class Transform1(object):


	def __init__(self):
		pass

	def __call__(self):
		pass

class Transform2(object):

	def __init__(self):
		pass
	
	def __call__(self):
		pass




	
import argparse
import numpy as np
import os
import pdb 

from library import models, architectures, utils
from library.architectures import ConvEncoder, ConvDecoder
from torch.utils import data
from torch import optim 
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="VAE model")
    parser.add_argument('--dataset', type=str, default='mnist',
                        metavar='N', help='use prestored datasets (default: mnist)')
    parser.add_argument('--n_epochs', type=int, default=1000,
                        metavar='N', help='number of epochs to train (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=32,
                        metavar='N', help='batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.0005,
                        metavar='N', help='learning rate for optimizer (default: 0.0005)')
    args = parser.parse_args()
    return args

def run_experiment(args):
    
    if args.dataset == 'mnist':
        path = '../data/mnist/'
    
    train_rawdata, val_rawdata = utils.img_to_npy(path=path,
                                                train=True,
                                                val_split_ratio=0.2
                                                )

    test_rawdata = utils.img_to_npy(path=path, train=False)

    train_data = utils.ImageData(rawdata=train_rawdata, transform=transforms.Normalize((0.1307,), (0.3081,) ))
    val_data = utils.ImageData(rawdata=val_rawdata, transform=transforms.Normalize((0.1307,), (0.3081,) ))
    test_data = utils.ImageData(rawdata=test_rawdata, transform=transforms.Normalize((0.1307,), (0.3081,) ))

    train_gen = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_gen = data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_gen = data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)
        

    # initialize model
    optim_dict = {'lr': args.lr, 'betas': (0.9, 0.99)}
    optimizer= optim.Adam
    model_dict = {'img_encoder': ConvEncoder(), 'img_decoder': ConvDecoder()}
    model = models.VaeGaussian(**model_dict)

    #model = models.VaeGaussian()
    #model = models.VaeGaussian(img_encoder=ConvEncoder(), img_decoder=ConvDecoder())
    #model = models.VaeGaussian(**model_dict)
    #model = models.VaeGaussian(img_encoder=ConvEncoder(), img_decoder=ConvDecoder())
    
    # train model
    for epoch in range(args.n_epochs):
        model._train(model=model, train_gen=train_gen, optim=optimizer, optim_dict=optim_dict)
        
        if epoch % args.val_steps == 0 and epoch > 0:
            model._validate(model=model, val_gen=val_gen)





if __name__ == "__main__":
    args = parse_args()
    run_experiment(args=args)
import argparse
import numpy as np
import os
import pdb 

from library import models, architectures, utils
from library.architectures import ConvEncoder, ConvDecoder, ConvEncoder28x28, ConvDecoder28x28
from torch.utils import data
from torch import optim 
from torchvision import transforms

#from pytorch_lightning.loggers import MLFlowLogger

def parse_args():
    parser = argparse.ArgumentParser(description="VAE model")
    parser.add_argument('--dataset', type=str, default='mnist',
                        metavar='N', help='use prestored datasets (default: mnist)')
    parser.add_argument('--n_epochs', type=int, default=10,
                        metavar='N', help='number of epochs to train (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=32,
                        metavar='N', help='batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.01,
                        metavar='N', help='learning rate for optimizer (default: 0.0005)')
    parser.add_argument('--lr_scheduler', type=float, default=True,
                        metavar='N', help='set learning rate scheduler (default: True)')
    parser.add_argument('--latent_dim', type=int, default=10,
                        metavar='N', help='latent_dim for VAE (default: 32')
    parser.add_argument('--in_channels', type=float, default=1,
                        metavar='N', help='number of channels RGB etc. (default:1)')
    parser.add_argument('--path', type=str, default='../logs/images/',
                        metavar='N', help='specify path where reconstructed and sampled images shall be stored')
    parser.add_argument('--experiment_name', type=str, default='mnist_vanilla_vae',
                        metavar='N', help='experiment name')
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

    #train_data = utils.ImageData(rawdata=train_rawdata, transform=transforms.Normalize((0.1307,), (0.3081,)))
    #val_data = utils.ImageData(rawdata=val_rawdata, transform=transforms.Normalize((0.1307,), (0.3081,)))
    #test_data = utils.ImageData(rawdata=test_rawdata, transform=transforms.Normalize((0.1307,), (0.3081,)))

    train_data = utils.ImageData(rawdata=train_rawdata)
    val_data = utils.ImageData(rawdata=val_rawdata)
    test_data = utils.ImageData(rawdata=test_rawdata)

    train_gen = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_gen = data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_gen = data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    # initialize model
    optim_dict = {'lr': args.lr, 'betas': (0.9, 0.99)}
    optimizer= optim.Adam
    latent_dim = args.latent_dim
    in_channels = args.in_channels
   
    
    model_dict = {'img_encoder': ConvEncoder28x28(in_channels=in_channels, latent_dim=latent_dim), 
                    'img_decoder': ConvDecoder28x28(in_channels=in_channels, latent_dim=latent_dim)}
    
    # for Gaussian VAE:
    #model = models.VaeGaussian(**model_dict)

    # For Info VAE:
    model = models.InfoVae(**model_dict)
    
    
    # train model
    for epoch in range(args.n_epochs):
        model._train(model=model, 
                    train_gen=train_gen, 
                    optimizer=optimizer,
                    optim_dict=optim_dict,
                    lr_scheduler=args.lr_scheduler)
        model._validate(model=model, train_gen=train_gen, val_gen=val_gen, epoch=epoch)
        model._sample_images(model=model,
                            val_gen=val_gen,
                            path=args.path,
                            epoch=epoch,
                            experiment_name=args.experiment_name)


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args=args)

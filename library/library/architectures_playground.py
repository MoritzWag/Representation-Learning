import torch
import torch.utils.data
import numpy as np
import unittest
import pdb

from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from torch import Tensor

torch.set_default_dtype(torch.float64)


class ConvEncoderAutomatic(nn.Module):
    """
    """

    def __init__(self,
                in_dims: float,
                in_channels: float,
                latent_dim: float,
                enc_padding: int,
                enc_stride: int,
                enc_kernel_size: int,
                enc_hidden_dims: int,
                **kwargs) -> None:
        super(ConvEncoderAutomatic, self).__init__()
            
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.in_dims = in_dims

        # Input check for padding
        #assertTrue(len(enc_hidden_dims) == len(enc_padding))

        # Input check for kernel size
        #assertTrue(len(enc_hidden_dims) == len(kernel_size))
        
        # Input check for stride
        #assertTrue(len(enc_hidden_dims) == len(stride))

        self.enc_hidden_dims = enc_hidden_dims
        self.enc_padding = enc_padding
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_stride

        modules = []

        output_dims = np.repeat(in_dims, len(self.enc_hidden_dims)+1)
        for i in range(len(self.enc_hidden_dims)):
            if i == 0:
                output_dims[i+1] = np.floor((output_dims[i+1] + 2*enc_padding[i] - enc_kernel_size[i])/enc_stride[i]) + 1
            else:
                output_dims[i+1] = np.floor((output_dims[i] + 2*enc_padding[i] - enc_kernel_size[i])/enc_stride[i]) + 1

        #assertNotIn(0, output_dims, 'H or W dimensions are 0 at layer {}. Please adjust kernel size, padding or stride to fix this'format(output_dims.index(0)))
        
        self.enc_output_dim = output_dims[-1]*output_dims[-1]

        for layer in range(len(self.enc_hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels = self.enc_hidden_dims[layer],
                                kernel_size = self.enc_kernel_size[layer],
                                 stride = self.enc_stride[layer],
                                  padding = self.enc_padding[layer]),
                    nn.BatchNorm2d(enc_hidden_dims[layer]),
                    nn.LeakyReLU()
                )
            )
            in_channels = self.enc_hidden_dims[layer]

        self.encoder = nn.Sequential(*modules)
        
    def forward(self, input: Tensor) -> Tensor:
        output = self.encoder(input)
        output = torch.flatten(output, start_dim=1)
        return output

class ConvDecoderAutomatic(nn.Module):
    """
    """
    
    def __init__(self,
                in_dims: float,
                in_channels: float,
                latent_dim: float,
                dec_hidden_dims: int,
                dec_padding: int,
                dec_stride: int,
                dec_kernel_size: int,
                dec_out_padding: int,
                enc_padding: int,
                enc_stride: int,
                enc_kernel_size: int,
                enc_hidden_dims,
                categorical_dim = None,
                **kwargs) -> None:
        super(ConvDecoderAutomatic, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels

        #self.in_dims = in_dims maybe use for later in first dec layer

        # Input check for padding
        #assertTrue(len(dec_hidden_dims) == len(dec_padding))

        # Input check for kernel size
        #assertTrue(len(dec_hidden_dims) == len(dec_kernel_size))
        
        # Input check for dec_stride
        #assertTrue(len(dec_hidden_dims) == len(dec_stride))

        # Input check for out padding
        #assertTrue(len(dec_hidden_dims)-1 == len(dec_out_padding))

        self.dec_padding = dec_padding
        self.dec_kernel_size = dec_kernel_size
        self.dec_stride = dec_stride
        self.dec_out_padding = dec_out_padding

        modules = []
        # Obtain output dimension of encoder
        output_dims = np.repeat(in_dims, len(enc_hidden_dims)+1)

        for i in range(len(enc_hidden_dims)):
            if i == 0:
                output_dims[i+1] = np.floor((output_dims[i+1] + 2*enc_padding[i] - enc_kernel_size[i])/enc_stride[i]) + 1
            else:
                output_dims[i+1] = np.floor((output_dims[i] + 2*enc_padding[i] - enc_kernel_size[i])/enc_stride[i]) + 1
        
        self.enc_output_dim = output_dims[-1]*output_dims[-1]
        
        # Check to adjust first layer in case categorical dimension exists
        if categorical_dim is None:
            self.decoder_input = nn.Linear(latent_dim, dec_hidden_dims[0] * self.enc_output_dim)
        else:
            self.categorical_dim = categorical_dim
            self.decoder_input = nn.Linear(latent_dim * categorical_dim, dec_hidden_dims[0] * self.enc_output_dim)

        self.dec_hidden_dims = dec_hidden_dims

        for layer in range(len(dec_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels = self.dec_hidden_dims[layer],
                                        out_channels = self.dec_hidden_dims[layer + 1],
                                        kernel_size = self.dec_kernel_size[layer],
                                        stride = self.dec_stride[layer],
                                        padding = self.dec_padding[layer],
                                        output_padding = self.dec_out_padding[layer]),
                    nn.BatchNorm2d(self.dec_hidden_dims[layer + 1]),
                    nn.LeakyReLU()
                )
            )
            in_channels = self.dec_hidden_dims[layer]

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(in_channels = self.dec_hidden_dims[-1],
                                                    out_channels = self.in_channels,
                                                    kernel_size = self.dec_kernel_size[-1],
                                                    stride = self.dec_stride[-1],
                                                    padding = self.dec_padding[-1]),
                                nn.BatchNorm2d(self.in_channels),
                                nn.Sigmoid())
    

    def forward(self, input) -> Tensor:
        x = self.decoder_input(input)
        x = x.view(-1,
         self.dec_hidden_dims[0],
          np.sqrt(self.enc_output_dim).astype(int),
           np.sqrt(self.enc_output_dim).astype(int)
         ) # will work only with quadratic 
        x = self.decoder(x)
        output = self.final_layer(x)
        return output
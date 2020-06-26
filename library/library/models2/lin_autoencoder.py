import pandas as pd
import numpy as np 
import pdb
import logging
import os
from library import architectures
from library.architectures import ConvEncoder28x28, ConvDecoder28x28
from library.models2.helpers import *

from library.models2 import base2
from library.models2.base2 import VaeBase
from library.models2.base2 import AutoencoderBase

import torch
import torch.utils.data
from torch.autograd import Variable
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils


class LinearAutoencoder(nn.Module):
    """
    """

    def __init__(self,
                **kwargs):
        super(LinearAutoencoder, self).__init__(
            **kwargs
        )
        
        self.latent_dim = self.img_encoder.latent_dim

    def _sample(self, num_samples):

        z = torch.randn(num_samples, 
                        self.latent_dim)
        
        if torch.cuda.is_available():
            z = z.cuda()

        samples = self.img_decoder(z.float())

        return samples

    def _embed(self, data):

        embedding = self.img_encoder(data.float())
        self.store_z = embedding

    def _loss_function(self, image=None, recon_image=None, *args, **kwargs):

        image_recon_loss = F.mse_loss(recon_image, image).to(torch.float64)
                
        return {'loss': image_recon_loss.to(torch.double)}
        
            

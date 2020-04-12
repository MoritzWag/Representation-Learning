import pandas as pd
import numpy as np 
import pdb
import logging
import os
from library import architectures
from library.architectures import ConvEncoder, ConvDecoder

#from library.models.base import VaeBase
from library.models2 import base2
from library.models2.base2 import VaeBase

import torch
import torch.utils.data
from torch.autograd import Variable
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
#from torchvision.utils import save_image, vutils



class VaeGaussian(VaeBase):
    """
    """
    def __init__(self, 
                **kwargs):
        super(VaeGaussian, self).__init__(
            **kwargs
        )
        self.latent_dim = self.img_encoder.latent_dim
        self.hidden_dim = self.img_encoder.hidden_dims
        self.mu = nn.Linear(self.hidden_dim[-1]*4, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dim[-1]*4, self.latent_dim)
        
    def _reparameterization(self, h_enc):
        """Reparameterization trick to sample from N(mu, var) from 
        N(0,1).

        Args:
            mu: {Tensor} Mean of the latent Gaussian [B x D]
            logvar: {Tensor} Standard Deviation of the latent Gaussian [B x D]
        Returns:
            {Tensor} {B x D}
        """
        ## Gaussian reparameterization: mu + epsilon*sigma
        mu = self.mu(h_enc)
        log_sigma = self.logvar(h_enc)

        std = torch.exp(0.5*log_sigma)
        eps = torch.randn_like(std)

        # store loss items
        self.loss_item['z_mean'] = mu
        self.loss_item['z_sigma'] = log_sigma

        return eps * std * mu


    def _loss_function(self, x: Tensor, recon_x: Tensor, z_mean: Tensor, z_sigma: Tensor):
        # latent loss: KL divergence
        def latent_loss(z_mean, z_sigma):
            return torch.mean(-0.5 * torch.sum(1 + z_sigma - z_mean ** 2 - z_sigma.exp(), dim=1), dim=0)
        
        # reconstruction loss 
        def reconstruction_loss(recon_x, x):
            return F.mse_loss(recon_x, x)
            #return F.binary_cross_entropy(recon_x, x)

        lat_loss = latent_loss(z_mean, z_sigma)
        recon_loss = reconstruction_loss(recon_x, x.float())

        kld_weight = 32 / 40000

        # should figure out how to properly weight the losses 
        loss = kld_weight * lat_loss + recon_loss

        return {'loss': loss.to(torch.double), 'lat_loss': lat_loss.to(torch.double), 'recon_loss': recon_loss.to(torch.double)}
        #return {'loss': loss.to(torch.DoubleTensor), 'lat_loss': lat_loss.to(torch.DoubleTensor), 'recon_loss': recon_loss.to(torch.DoubleTensor)}
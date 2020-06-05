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

import torch
import torch.utils.data
from torch.autograd import Variable
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils


class Autoencoder(nn.Module):
    """
    """

    def __init__(self,
                l1_reg: bool = False,
                **kwargs):
        super(Autoencoder, self).__init__(
            **kwargs
        )
        self.latent_dim = self.img_encoder.latent_dim
        self.hidden_dim = self.img_encoder.enc_hidden_dims
        self.output_dim = self.img_encoder.enc_output_dim

        # Define the affine linear transformations from laster layer of encoder to space of latents
        self.affine_lin = nn.Linear(self.hidden_dim[-1] * self.output_dim, self.latent_dim)
        self.l1_reg = l1_reg
        self.l1_reg = l1_reg

    def _reparameterization(self, h_enc):
        
        """Merely performs a affine linear transformation from the encoder 
        output to the latent dimensions

        Args:
            h_enc: output from encoder (flattend tensor)
        Returns:
            {Tensor} {B x D}
        """
        ## Gaussian reparameterization: mu + epsilon*sigma
        latent_code = self.affine_lin(h_enc)

        # store latent code
        self.loss_item['code'] = latent_code

        return latent_code
    
    def _parameterize(self, h_enc, img=None, attrs=None):
        pass

    def _sample(self, num_samples):
        """Samples from the latent space and return the corresponding
        image space map.

        Should be defined as a one-step or multistep sampling scheme depending on the stochastic nodes in the architecture
        
        Args:
            num_samples {int}: number of samples
        Returns:
            {Tensor}
        """ 
        z = torch.randn(num_samples, 
                        self.latent_dim)
        
        if torch.cuda.is_available():
            z = z.cuda()

        samples = self.img_decoder(z.float())

        return samples

    def _embed(self, data):
        """
        """
        #x = self.resnet(data.float())
        if torch.cuda.is_available():
            data = data.cuda()
        embedding = self.img_encoder(data.float())
        mu = self.mu(embedding)
        logvar = self.logvar(embedding)
        z = self._reparameterization(embedding)

        return mu, logvar, embedding

    def _loss_function(self, image=None, recon_image=None, code=None, *args, **kwargs):

        image_recon_loss = F.mse_loss(recon_image, image).to(torch.float64)
        loss = image_recon_loss

        if self.l1_reg == True:
            loss = image_recon_loss + torch.mean(torch.sum(code.abs(), 1))
        
        latent_loss = torch.tensor(0)
        
        return {'loss': loss.to(torch.double), 'latent_loss': latent_loss.to(torch.double), 
                    'image_recon_loss': image_recon_loss.to(torch.double)}
        
            

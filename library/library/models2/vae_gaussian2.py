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
#from torchsummary import summary

torch.set_default_dtype(torch.float64)

class VaeGaussian(nn.Module):
    """
    """
    def __init__(self,
                trial=None,
                loss_type = "l2",
                kld_weight: float = 32/40000, 
                **kwargs):
        super(VaeGaussian, self).__init__(
            **kwargs
        )

        self.latent_dim = self.img_encoder.latent_dim
        self.hidden_dim = self.img_encoder.enc_hidden_dims
        self.output_dim = self.img_encoder.enc_output_dim
        self.loss_type = loss_type
        self.kld_weight = kld_weight
        # Define the affine linear transformations from laster layer of encoder to space of parametrization
        self.mu = nn.Linear(self.hidden_dim[-1] * self.output_dim, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dim[-1] * self.output_dim, self.latent_dim)
        
        try:
            self.attr_mu = nn.Linear(50, self.text_encoder.num_attr) #hard-coded
            self.attr_logvar = nn.Linear(50, self.text_encoder.num_attr) #hard-coded
        except:
            pass

    # here I need some check whether reparameterization for attr or image 
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
        logvar = self.logvar(h_enc)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        # store loss items
        self.loss_item['mu'] = mu
        self.loss_item['logvar'] = logvar
        z = eps * std + mu
        
        return z
    
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

    def _embed(self, data, return_latents=False):
        """
        """
        embedding = self.img_encoder(data.float())
        mu = self.mu(embedding)
        logvar = self.logvar(embedding)
        z = self._reparameterization(embedding)

        if return_latents == True:
            return z

        self.store_z = z
        self.mu_hat = z.transpose(dim0 = 0, dim1 = 1).mean(dim = 1)
        self.sigma_hat = z.transpose(dim0 = 0, dim1 = 1).var(dim = 1).sqrt()

        

    def _parameterize(self, h_enc, img=None, attrs=None):

        if img:
            mu = self.mu(h_enc)
            log_sigma = self.logvar(h_enc)
        else:
            mu = self.attr_mu(h_enc)
            log_sigma = self.attr_mu(h_enc)
        
        return mu, log_sigma

    def _mm_reparameterization(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def _loss_function(self, image=None, text=None, recon_image=None, 
                        recon_text=None, mu=None, logvar=None):
        
        if self.loss_type == "l2":

            if recon_image is not None and image is not None:
                image_recon_loss = F.mse_loss(recon_image, image).to(torch.float64)
            
            if recon_text is not None and text is not None:
                text_recon_loss = F.nll_loss(recon_text, text.to(torch.long)).to(torch.float64)
        else:

            if recon_image is not None and image is not None:
                image_recon_loss = F.l1_loss(recon_image, image).to(torch.float64)
            
            if recon_text is not None and text is not None:
                text_recon_loss = F.nll_loss(recon_text, text.to(torch.long)).to(torch.float64)

        latent_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        if recon_text is not None and text is not None:
            loss = kld_weight * latent_loss + image_recon_loss + text_recon_loss 
            return {'loss': loss.to(torch.double), 'latent_loss': latent_loss.to(torch.double), 
                    'image_recon_loss': image_recon_loss.to(torch.double), 'text_recon_loss': text_recon_loss.to(torch.double)}

        else:
            loss = self.kld_weight * latent_loss + image_recon_loss 

            return {'loss': loss.to(torch.double), 'latent_loss': latent_loss.to(torch.double), 
                    'image_recon_loss': image_recon_loss.to(torch.double)}
        

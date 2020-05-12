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


class VaeBeta(nn.Module):
    """
    """

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                beta: int = 4,
                max_capacity: int = 25, # essentially the distance that we allow the prior and the posterior to have; 0 means: no distance is allowed/wanted
                restrict_capacity = False,
                capacity_max_iter: int = 1e5,
                **kwargs):
        super(VaeBeta, self).__init__(
            **kwargs
        )
        self.latent_dim = self.img_encoder.latent_dim
        self.hidden_dim = self.img_encoder.hidden_dims
        self.mu = nn.Linear(self.hidden_dim[-1]*4, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dim[-1]*4, self.latent_dim)
        self.beta = beta
        self.c_max = torch.Tensor([max_capacity])
        self.c_stop_iter = capacity_max_iter
        self.restrict_capacity = restrict_capacity

        try:
            self.attr_mu = nn.Linear(50, self.text_encoder.num_attr)
            self.attr_logvar = nn.Linear(50, self.text_encoder.num_attr)
        except:
            pass

        assert self.beta >= 0

    # here I need some check whether reparameterization for attr or image 
    def _reparameterization(self, h_enc):
        #pdb.set_trace()
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

        return eps * std * mu
    
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
                        recon_text=None, mu=None, logvar=None, *args, **kwargs):
        
        self.num_iter += 1

        if recon_image is not None and image is not None:
            image_recon_loss = F.mse_loss(recon_image, image).to(torch.float64)
        
        if recon_text is not None and text is not None:
            text_recon_loss = F.nll_loss(recon_text, text.to(torch.long)).to(torch.float64)
        
        #latent_loss = torch.mean(-0.5 * torch.sum(1 + logvar + mu ** 2 - logvar.exp(), dim=1), dim=0)
        latent_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        kld_weight = 32 / 40000
        #loss = kld_weight * latent_loss + image_recon_loss + text_recon_loss
        #pdb.set_trace()
        
        if self.restrict_capacity == False:

            if recon_text is not None and text is not None:
                loss = kld_weight * self.beta * latent_loss + image_recon_loss + text_recon_loss 
                return {'loss': loss.to(torch.double), 'latent_loss': latent_loss.to(torch.double), 
                        'image_recon_loss': image_recon_loss.to(torch.double), 'text_recon_loss': text_recon_loss.to(torch.double)}
                        
            else:
                loss = kld_weight * self.beta * latent_loss + image_recon_loss 

                return {'loss': loss.to(torch.double), 'latent_loss': latent_loss.to(torch.double), 
                        'image_recon_loss': image_recon_loss.to(torch.double)}
        
        # beta-VAE with increased capacity with every iteration
        else:
            self.c_max = self.c_max.to(input.device)
            capacity = torch.clamp(self.c_max/self.c_stop_iter * self.num_iter, 0, self.c_max.data[0])

            if recon_text is not None and text is not None:
                loss = image_recon_loss + text_recon_loss + self.beta * kld_weight * (kld_loss - capacity).abs()

                return {'loss': loss.to(torch.double), 'latent_loss': latent_loss.to(torch.double), 
                        'image_recon_loss': image_recon_loss.to(torch.double), 'text_recon_loss': text_recon_loss.to(torch.double)}
                        
            else:
                loss = image_recon_loss + self.beta * kld_weight * (kld_loss - capacity).abs()

                return {'loss': loss.to(torch.double), 'latent_loss': latent_loss.to(torch.double), 
                        'image_recon_loss': image_recon_loss.to(torch.double)}
        
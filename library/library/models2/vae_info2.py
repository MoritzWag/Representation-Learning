import pandas as pd
import numpy as np 
import pdb
import logging
import os
from library import architectures
from library.architectures import ConvEncoder, ConvDecoder

from library.models.base import VaeBase


import torch
import torch.utils.data
from torch.autograd import Variable
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
#from torchvision.utils import save_image, vutils



class InfoVae(VaeBase):
    """
    """
    
    def __init__(self,
            alpha = -0.5,
            beta = 5.0,
            reg_weight = 100,
            kernel_type = 'imq',
            latent_var = 2.,
            **kwargs):
        super(InfoVae, self).__init__(
            **kwargs
        )

        self.latent_dim = self.img_encoder.latent_dim
        self.hidden_dim = self.img_encoder.hidden_dims
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var

        assert alpha <= 0, "alpha must be negative or zero"
        self.alpha = alpha
        self.beta = beta

        self.mu = nn.Linear(self.hidden_dim[-1]*4, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dim[-1]*4, self.latent_dim)
    
    def _reparameterization(self, henc):
       
        mu = self.mu(henc)
        logvar = self.logvar(henc)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        self.loss_item['mu'] = mu
        self.loss_item['logvar'] = logvar

        z = eps * std + mu 
        self.loss_item['z'] = z

        return z
    
    def _loss_function(self, x, recon_x, z, mu, logvar):

        batch_size = 32
        bias_corr = batch_size * (batch_size - 1)
        kld_weight = 32 / 40000

        recon_loss = F.mse_loss(recon_x, x)
        mmd_loss = self.compute_mmd(z).to(torch.double)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        loss = self.beta* recon_loss + \
                (1. -  self.alpha) * kld_weight * kld_loss + \
                (self.alpha + self.reg_weight - 1.) / bias_corr * mmd_loss
        
        return {'loss': loss, 'lat_loss': kld_loss, 'recon_loss': recon_loss, 'mmd_loss': mmd_loss}
    
    def compute_kernel(self,
                       x1: Tensor,
                       x2: Tensor) -> Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result


    def compute_rbf(self,
                    x1: Tensor,
                    x2: Tensor,
                    eps: float = 1e-7) -> Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self,
                               x1: Tensor,
                               x2: Tensor,
                               eps: float = 1e-7) -> Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by
                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z: Tensor) -> Tensor:
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + \
              z__kernel.mean() - \
              2 * priorz_z__kernel.mean()
        return mmd


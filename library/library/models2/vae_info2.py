import pandas as pd
import numpy as np 
import pdb
import logging
import os
from library import architectures
from library.architectures import ConvEncoder28x28, ConvDecoder28x28
from library.models2.helpers import *

from library.models2.base2 import VaeBase


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

        try:
            self.attr_mu = nn.Linear(50, self.text_encoder.num_attr)
            self.attr_logvar = nn.Linear(50, self.text_encoder.num_attr)
        except:
            pass
    
    def _reparameterization(self, h_enc):
       
        mu = self.mu(h_enc)
        logvar = self.logvar(h_enc)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        self.loss_item['mu'] = mu
        self.loss_item['logvar'] = logvar

        z = eps * std + mu 
        self.loss_item['z'] = z

        return z
    
    def _parameterize(self, h_enc, img=None, attrs=None):

        if img:
            mu = self.mu(h_enc)
            log_sigma = self.logvar(h_enc)
        else:
            mu = self.attr_mu(h_enc)
            log_sigma = self.logvar(h_enc)
        
        return mu, log_sigma
    
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

    def _mm_reparameterization(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return eps * std + mu
    
    def _loss_function(self, image=None, text=None, recon_image=None,
                        recon_text=None, mu=None, logvar=None, z=None, **kwargs):
        
        batch_size = 32
        bias_corr = batch_size * (batch_size - 1)

        if image.shape[1] == 3:
            kld_weight = 32 / 400000
        else:
            kld_weight = 32 / 40000

        if recon_image is not None and image is not None:
            image_recon_loss = F.mse_loss(recon_image, image).to(torch.float64)
        
        if recon_text is not None and text is not None:
            text_recon_loss = F.nll_loss(recon_text, text.to(torch.long)).to(torch.float64)
        
        latent_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        mmd_loss = self.compute_mmd(z).to(torch.double)
        
        if recon_text is not None and text is not None:
            loss = self.beta*(image_recon_loss + text_recon_loss) + \
                    (1. - self.alpha) * kld_weight * latent_loss + \
                    (self.alpha + self.reg_weight -1.)/bias_corr * mmd_loss
            return {'loss': loss.to(torch.double), 'latent_loss': latent_loss.to(torch.double),
                    'image_recon_loss': image_recon_loss.to(torch.double), 'text_recon_loss': text_recon_loss.to(torch.double),
                    'mmd_loss': mmd_loss.to(torch.double)}
        else: 
            loss = self.beta*(image_recon_loss) + \
                    (1. - self.alpha) * kld_weight * latent_loss + \
                    (self.alpha + self.reg_weight -1.)/bias_corr * mmd_loss
            return {'loss': loss.to(torch.double), 'latent_loss': latent_loss.to(torch.double),
                    'image_recon_loss': image_recon_loss.to(torch.double), 'mmd_loss': mmd_loss.to(torch.double)}

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


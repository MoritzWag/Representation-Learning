import pandas as pd 
import numpy as np 
import pdb 
import logging
import os 

from library import architectures

from library.models2.helpers import * 
from library.models2 import base2
from library.models2.base2 import VaeBase

import torch 
import torch.utils.data
from torch.autograd import Variable
from torch import nn, optim, Tensor
from torch.nn import functional as F 
from torchvision import datasets,  transforms
import torchvision.utils as vutils 

torch.set_default_dtype(torch.float64)



class JointVae(nn.Module):
    """
    """
    num_iter = 1
    def __init__(self, 
                latent_min_capacity: float = 0.,
                latent_max_capactiy: float = 25.,
                latent_gamma: float = 30,
                latent_num_iter: int = 25000,
                categorical_min_capacity: float = 0., 
                categorical_max_capacity: float = 25.,
                categorical_gamma: float = 30.,
                categorical_num_iter: int = 25000,
                temperature: float = 0.5,
                anneal_rate: float = 3e-5,
                anneal_interval: int = 100,
                alpha: float = 30.,
                **kwargs):
        super(JointVae, self).__init__(
            **kwargs
        )

        self.latent_dim = self.img_encoder.latent_dim
        self.hidden_dim = self.img_encoder.enc_hidden_dims
        self.output_dim = self.img_encoder.enc_output_dim

        self.mu = nn.Linear(self.hidden_dim[-1] * self.output_dim, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dim[-1] * self.ouput_dim, self.latent_dim)
        self.cat = nn.Linear(self.hidden_dim[-1] * self.output_dim, self.categorical_dim)

        self.temperature = temperature
        self.min_temperature = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval 
        self.alpha = alpha 

        self.latent_min_capacity = latent_min_capacity
        self.latent_max_capacity = latent_max_capacity 

        self.categorical_min_capacity = categorical_min_capacity
        self.categorical_max_capacity = categorical_max_capacity

        self.latent_gamma = latent_gamma
        self.categorical_gamma = categorical_gamma

        self.latent_num_iter = latent_num_iter 
        self.categorical_num_iter = categorical_num_iter
    
    def _reparameterization(self, h_enc, constant=1e-7):
        """
        """
        mu = self.mu(h_enc)
        logvar = self.logvar(h_enc)
        logits = self.cat(h_enc)
        logits = logits.view(-1, self.categorical_dim)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        self.loss_item['mu'] = mu
        self.loss_item['logvar'] = logvar
        self.loss_item['logits'] = logits

        z = eps * std + mu 

        uniform_samples = torch.randn_like(q)
        gumbel_samples = - torch.log(-torch.log(uniform_samples + constant) + constant)

        if not self.training:
            probs = F.softmax((logits + gumbel_samples), dim=-1)
        else:
            probs = F.softmax((logits + gumbel_samples / self.temp), dim=-1)

        probs = probs.view(-1, self.categorical_dim)

        return torch.cat([z, probs], dim=1)

    def _sample(self, num_samples):
        """Samples from the latent space and returns 
        the corresponding image space map.
        """
        z = torch.randn(num_samples,
                        self.latent_dim)
        
        np_y = np.zeros((num_samples, self.categorical_dim), dtype=np.float32)
        np_y[range(num_samples), np.random.choice(self.categorical_dim, num_samples)] = 1
        np_y = np.reshape(np_y, [num_samples, self.categorical_dim])
        q = torch.from_numpy(np_y)

        z = torch.cat([z, q], dim=1)

        
        if torch.cuda.is_available():
            z = z.cuda()
        
        samples = self.img_encoder(z.float())

        return samples

    def _embed(self, data, constant=1e-7):
        """
        """
        embedding = self.img_encoder(data.float())
        mu = self.mu(embedding)
        logvar = self.logvar(embedding)
        logits = self.cat(embedding)
        probs = F.softmax(logit, dim=-1)

        probs = probs.view(-1, self.categorical_dim)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = eps * std + mu 

        z_probs = torch.cat([z, probs], dim=1)
        self.store_z_probs = z_probs
        self.store_individual_z = z
        self.store_probs = probs
        self.mu_hat = z.transpose(dim0=0, dim1=2).transpose(dim0=0, dim1=1).mean(dim=2)
        self.sigma_hat = z.transpose(dim0=0, dim1=2).transpose(dim0=0, dim1=1).var(dim=2).sqrt()


    
    def _parameterize(self, h_enc, img=None, attrs=None):
        pass 
    
    def _loss_function(self, image=None, text=None, recon_image=None,
                    recon_text=None, mu=None, logvar=None, logits=None, *args, **kwargs):

        q_p = F.softmax(logits, dim=1)

        kld_weight = 32 / 40000
        if batch_idx % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(- self.anneal_rate * batch_idx), self.min_temp)
        
        image_recon_loss = F.mse_loss(recon_image, image, reduction='mean')

        categorical_curr = (self.categorical_max_capacity - self.categorical_min_capacity) * \
                            self.num_iter / float(self.categorical_num_iter) + self.categorical_min_capacity
        categorical_curr = min(categorical_curr, np.log(self.categorical_dim))

        eps = 1e-7

        # entropy of logits 
        h1 = q_p * torch.log(q_p + eps)
        # cross entropy with categorical distribution 
        h2 = q_p * np.log(1. / self.categorical_dim + eps)

        kld_categorical_loss = torch.mean(torch.sum(h1 - h2, dim=1), dim=0)

        # Compute continouos loss 
        latent_curr = (self.latent_max_capacity - self.latent_min_capacity) * \
                    self.num_iter / float(self.latent_num_iter) + self.latent_min_capacity
        
        latent_curr = min(latent_curr, self.latent_max_capacity)

        kld_latent_loss = torch.mean( - 0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        capacity_loss = self.categorical_gamma * torch.abs(categorical_curr - kld_categorical_loss) + \
                        self.latent_gamma * torch.abs(latent_curr - kld_latent_loss)
        
        loss = self.alpha * image_recon_loss + kld_weight * capacity_loss

        if self.training:
            self.num_iter += 1
        
        return {'loss': loss, 'image_recon_loss': image_recon_loss, 'capacity_loss': capacity_loss}
    
    
        

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

torch.set_default_dtype(torch.float64)

class GaussmixVae(nn.Module):
    """
    """

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(
    self,
    temperature,
    anneal_rate,
    temperature_bound,
    decrease_temp,
    anneal_interval,
    kld_weight,
    cont_weight,
    cat_weight,
    trial=None,
    **kwargs):
        super(GaussmixVae, self).__init__(
            **kwargs
        )

        self.latent_dim = self.img_encoder.latent_dim
        self.hidden_dim = self.img_encoder.enc_hidden_dims
        self.output_dim = self.img_encoder.enc_output_dim
        self.categorical_dim = self.img_decoder.categorical_dim

        
        if trial is not None:
            self.temp = trial.suggest_int("temperature", 1, 40, step=10)
            self.temp_bound = temperature_bound
            self.anneal_rate = trial.suggest_float("anneal_rate", 0.1, 0.5, step=0.1)
            self.anneal_interval = anneal_interval
            self.kld_weight = kld_weight
            self.cont_weight = trial.suggest_int("cont_weight", 1, 3, step=1)
            self.cat_weight = trial.suggest_int("cat_weight", 1, 3, step=1)
            self.decr = decrease_temp
        else:
            self.temp = temperature
            self.temp_bound = temperature_bound
            self.anneal_rate = anneal_rate
            self.anneal_interval = anneal_interval
            self.kld_weight = kld_weight
            self.cont_weight = cont_weight
            self.cat_weight = cat_weight
            self.decr = decrease_temp



        self.mu = nn.Linear(
            self.hidden_dim[(-1)] * self.output_dim, self.latent_dim * self.categorical_dim)
        self.logvar = nn.Linear(
            self.hidden_dim[(-1)] * self.output_dim, self.latent_dim * self.categorical_dim)
        self.cat = nn.Linear(
            self.hidden_dim[(-1)] * self.output_dim, self.categorical_dim)

    # here I need some check whether reparameterization for attr or image
    def _reparameterization(self, h_enc: Tensor, constant: float=1e-07, *args, **kwargs):
        """
        Gumbel-Softmax reparametrization trick and followed by a normal reparametrization trick
        Args:
            h_enc: {Tensor} Last layer (flattend) of forward pass [B x D]
            eps: {Scalar} Machine Epsilon
        Returns:
            {Tensor} [B x D] 
        """
        mu = self.mu(h_enc)
        mu = mu.view(-1, self.categorical_dim, self.latent_dim)
        logvar = self.logvar(h_enc)
        logvar = logvar.view(-1, self.categorical_dim, self.latent_dim)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        logits = self.cat(h_enc)
        uniform_samples = torch.rand_like(logits)
        gumbel_samples = -torch.log(-torch.log(uniform_samples + constant) + constant)

        if not self.training:
            probs = F.softmax((logits + gumbel_samples), dim=(-1)) # temperature is not included during test time
        else:
            probs = F.softmax(((logits + gumbel_samples) / self.temp), dim=(-1))

        self.mu_hat_loss = z.transpose(dim0=0, dim1=2).transpose(dim0=0, dim1=1).mean(dim=2)

        probs = probs.unsqueeze(1)

        mixtures = torch.matmul(probs, z)
        mixtures = mixtures.squeeze(1)

        self.loss_item['mu'] = mu
        self.loss_item['logvar'] = logvar
        self.loss_item['logits'] = logits

        return mixtures
    
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
        mu = self.mu(embedding).view(-1, self.categorical_dim, self.latent_dim)
        logvar = self.logvar(embedding).view(-1, self.categorical_dim, self.latent_dim)
        probs = F.softmax(self.cat(embedding), dim=-1)
        z = torch.randn_like(logvar) * torch.exp(0.5 * logvar) + mu
        probs_ = probs.unsqueeze(1)
        mixtures = torch.matmul(probs_, z).squeeze(1)

        if return_latents == True:
            return mixtures

        # Store variables
        self.store_probs = probs
        self.store_individual_z = z
        self.mu_hat = z.transpose(dim0=0, dim1=2).transpose(dim0=0, dim1=1).mean(dim=2)
        self.sigma_hat = z.transpose(dim0=0, dim1=2).transpose(dim0=0, dim1=1).var(dim=2).sqrt()
        self.store_z = mixtures
        
    def _parameterize(self, h_enc, img=None, attrs=None):
        
        if img:
            mu = self.mu(h_enc)
            mu = mu.view(-1, self.categorical_dim, self.latent_dim)
            logvar = self.logvar(h_enc)
            logvar = logvar.view(-1, self.categorical_dim, self.latent_dim)

            logits = self.cat(h_enc)
            probs = F.softmax((logits), dim=(-1))

            probs = probs.unsqueeze(1)

            mixtures_mu = torch.matmul(probs, mu)
            mixtures_mu = mixtures_mu.squeeze(1)

            mixtures_logvar = torch.matmul(probs, logvar)
            mixtures_logvar = mixtures_logvar.squeeze(1)
        else:
            return 

        return mixtures_mu, mixtures_logvar

    def _loss_function(
        self,
        image=None,
        text=None,
        recon_image=None,
        recon_text=None,
        mu=None,
        logvar=None,
        logits=None,
        constant=1e-07,
        *args,
        **kwargs):

        if self.training:
            self.num_iter += torch.tensor(1).float()

        if recon_image is not None:
            if image is not None:
                image_recon_loss = F.mse_loss(recon_image, image).to(torch.float64)

        probs = F.softmax(logits/self.temp, dim=(-1))
        ent = torch.sum((probs * torch.log(probs + constant)), dim=1)
        c_ent = -torch.sum((probs * np.log(1.0 / self.categorical_dim + constant)), dim=1)
        kld_categorical = torch.mean(c_ent - ent)

        mu_c = torch.cat([self.mu_hat_loss.unsqueeze(0)] * image.size()[0], 0)
        kld_gaussmix = -0.5 * (1 - logvar.exp() + logvar - (mu - mu_c) ** 2)
        kld_gaussmix = torch.matmul(probs.unsqueeze(1), kld_gaussmix).squeeze(1)
        kld_gaussmix = torch.mean(torch.sum(kld_gaussmix, dim=1), dim=0)

        if self.num_iter % self.anneal_interval == 0:
            if self.training:
                if self.decr:
                    self.temp = np.maximum(self.temp * np.exp(-self.anneal_rate), self.temp_bound)
                else:
                    self.temp = np.minimum(self.temp * 1/np.exp(-self.anneal_rate), self.temp_bound)
        
        loss = image_recon_loss + self.kld_weight * (self.cont_weight*kld_gaussmix + self.cat_weight*kld_categorical)
        
        return {'loss':loss.to(torch.double), 
         'latent_loss':kld_gaussmix.to(torch.double), 
         'kld_categorical_loss':kld_categorical.to(torch.double), 
         'image_recon_loss':image_recon_loss.to(torch.double), 
         'temperature':torch.tensor(self.temp).to(torch.double), 
         'categorical_entropy':-torch.mean(ent).to(torch.double)
         }
        

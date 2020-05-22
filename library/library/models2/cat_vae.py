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
from torch.distributions import Dirichlet
from torch.distributions import Categorical


torch.set_default_dtype(torch.float64)

class CatVae(nn.Module):
    """ Implementation of CatVae paper: 

    Args:
        categorical_dim: {int} number of categories
        latent_dim: {int} number of latent dimensions to sample from 
        hidden_dim: {int} number of hidden dims
        ouput_dim: {int} flattened output dimension of the output dimension
        categorical_dim: {}
        temperature: {}
        anneal_rate: {}
        anneal_interval: {}
        alpha: {}
        probability: {}
    
    Returns:

    """
    def __init__(self,
                categorical_dim: int = 10,
                temperature: float = 0.5,
                anneal_rate: float = 3e-5,
                anneal_interval: int = 100,
                alpha: float = 30,
                probability = True,
                **kwargs):
        super(CatVae, self).__init__(
            **kwargs
        )
        self.latent_dim = self.img_encoder.latent_dim
        self.hidden_dim = self.img_encoder.enc_hidden_dims
        self.output_dim = self.img_encoder.enc_output_dim
        self.categorical_dim = self.img_decoder.categorical_dim # Number of categories
        self.temp = temperature # Temperature defines how close our gumbel-softmax is to gumbel-argmax (temp = 0)
        self.min_temp = temperature # Limit to which the temperature is annealed to
        self.anneal_rate = anneal_rate # Rate by which temperature is annealed to 0
        self.anneal_interval = anneal_interval # The intervall in which the temperature is annealed to 0
        self.alpha = alpha # Weight of the reconstruction loss
        self.probability = probability # Should probabilities or one hot encodings be sampled
        # The number of latent variables is defined by the product of the number
        # of categorical distributions K and the number of latent variables L
        # So, for instance we have L = 10 latent variables with each a categorical 
        # distribution 

        self.latents = nn.Linear(self.hidden_dim[-1]*self.output_dim, self.latent_dim * self.categorical_dim)

        try:
            self.attr_mu = nn.Linear(50, self.text_encoder.num_attr)#hard-coded
            self.attr_logvar = nn.Linear(50, self.text_encoder.num_attr)#hard-coded
        except:
            pass
     
    def _reparameterization(self, h_enc: Tensor, eps: float = 1e-7, *args, **kwargs):
        """
        Gumbel-Softmax reparametrization trick 
        Args:
            h_enc: {Tensor} Last layer (flattend) of forward pass [B x D]
            eps: {Scalar} Machine Epsilon
        Returns:
            {Tensor} [B x D] 
        """

        # Aff. lin. trafo. of the last encoder layer to latent space
        z = self.latents(h_enc)
        # Transformation of tenfor from [Batch x Latents*Cats] to [Batch x Latents x Categories]
        z = z.view(-1, self.latent_dim, self.categorical_dim)
        self.loss_item['code'] = z 

        # Sample from Gumbel Distribution
        uniform_samples = torch.rand_like(z)
        gumbel_samples = - torch.log(- torch.log(uniform_samples + eps) + eps)

        # Apply Gumbel samples to softmax function
        output = F.softmax((z + gumbel_samples) / self.temp, dim = -1)
        # Flatten the probabilities 
        output_flattend = output.view(-1, self.latent_dim * self.categorical_dim)

        # store probabilities in tensor shape as loss item
        #self.loss_item['code'] = output

        return output_flattend
    
    def _sample(self, num_samples, probability = True, *args, **kwargs):
        """Samples from the latent space and return the corresponding
        image space map.

        Should be defined as a one-step or multistep sampling scheme depending on the stochastic nodes in the architecture
        
        Args:
            num_samples {int}: number of samples
            probability {bool}: if probabilities should be used or one-hot-categorical vectors
        Returns:
            {Tensor}
        """ 

        if probability == True:
            # Sample probabilities from simplex modeled with dirichlet distribution
            dirichlet_dist = Dirichlet((0 - 2) * torch.rand(num_samples, self.latent_dim, self.categorical_dim) + 2)
            z = dirichlet_dist.sample()
            
        else:
            z = OneHotCategorical(torch.rand(num_samples, self.latent_dim, self.categorical_dim)).sample()
        
        z = z.view(num_samples, self.latent_dim * self.categorical_dim)

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
        z = self._reparameterization(embedding)

        return embedding, z 

    def _parameterize(self):
        pass

    def _loss_function(self, image=None, text=None, recon_image=None, 
                        recon_text=None, code=None, batch_idx=None, *args, **kwargs):

        # Compute the reconstruction loss
        if recon_image is not None and image is not None:
            image_recon_loss = F.mse_loss(recon_image, image).to(torch.float64)

        #try:
        #    #pdb.set_trace()
        #    #batch_idx = kwargs['batch_idx']
        #    batch_idx = batch_idx
        #
        #    # Anneal the temperature at regular intervals
        #    if self.batch_idx % self.anneal_interval == 0 and self.training:
        #        self.temp = np.maximum(self.temp * np.exp(- self.anneal_rate * batch_idx),
        #        self.min_temp)
        #except:
        #    pass


        # Compute the KL-Divergence between the categorical distribution and  
        # the prior of a categorical distribution with equal probabilites.
        # KL divergence is the Entropy of the posterior + the Cross-Entropy 
        # between the prior and the posterior w.r.t. posterior
        # Compute the Entropy
        eps = 1e-7

        code = F.softmax(code, dim=-1)
        ent = code * torch.log(code + eps)

        # Compute the Cross-Entropy
        c_ent = code * np.log(1. / self.categorical_dim + eps)

        # Compute KL-D 
        latent_loss = torch.mean(
            torch.sum(
                ent - c_ent,
                dim = (1, 2)
            ),
            dim = 0
        )

        kld_weight = 1.2 #seems pretty arbitrary....
        loss = kld_weight * latent_loss + self.alpha * image_recon_loss 

        return {'loss': loss.to(torch.double), 'latent_loss': latent_loss.to(torch.double), 
                'image_recon_loss': image_recon_loss.to(torch.double)}
        

import pandas as pd
import numpy as np 
import pdb
import logging
import os

from library.architectures import prior_experts, CustomizedResNet101
from library.visualizer import Visualizer
from abc import ABC, abstractmethod

import torch
import torch.utils.data
from torch.nn import functional as F
from torch import nn, optim, Tensor
import torchvision.utils as vutils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_default_dtype(torch.float64)

class ReprLearner(Visualizer):
    """Representation Learner with PyTorch base functionality.
    """

    def __init__(self, **kwargs):
        super(ReprLearner, self).__init__(**kwargs)
        self.loss_item = {}
    
    def _generate(self, x):
        """Given an input image x, returns the reconstructed image

        Args:

            model: 
            x: {Tensor} [B x C x H x W]
        Returns:
            {Tensor} [B x C H x W]
        """

        if torch.cuda.is_available():
            x = x.cuda()
        return self.forward(x.float())

    @abstractmethod
    def _sample(self):
        """Samples from the latent space and return the corresponding
        image space map.

        Should be defined as a one-step or multistep sampling scheme depending on the stochastic nodes in the architecture
        
        Args:
            num_samples {int}: number of samples
        Returns:
            {Tensor}
        """ 
        pass

    @abstractmethod
    def _embedding(self):
        """
        """
        pass


###########################################
#
# Unimodal vae image base learner
#
###########################################



class MMVaeBase(ReprLearner):
    """Create a Base class for Multimodal VAE which inherits from ReprLearner.
    """     
    def __init__(self,
            img_encoder,
            img_decoder,
            text_encoder,
            text_decoder,
            expert,
            **kwargs):
        super(MMVaeBase, self).__init__(**kwargs)
        self.img_encoder = img_encoder
        self.img_decoder = img_decoder
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.expert = expert

    def _reparameterization(self, x):
        pass

    def forward(self, image=None, attrs=None):
        assert image is not None or attrs is not None

        batch_size = image.size(0) if image is not None else attrs.size(0)
        
        mu, logvar = prior_experts((1, batch_size, 10))

        if image is not None and attrs is not None:
            img_henc = self.img_encoder(image)
            attrs_henc = self.text_encoder(attrs.to(torch.int64))

            image_mu, image_logvar = self._parameterize(img_henc, img=True)
            mu = torch.cat((mu, image_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, image_logvar.unsqueeze(0)), dim=0)

            attr_mu, attr_logvar = self._parameterize(attrs_henc, attrs=True)
            mu = torch.cat((mu, attr_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, attr_logvar.unsqueeze(0)), dim=0)

        elif image is not None:
            img_henc = self.img_encoder(image)
            image_mu, image_logvar = self._parameterize(img_henc, img=True)
            mu = torch.cat((mu, image_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, image_logvar.unsqueeze(0)), dim=0)

        elif attrs is not None:
            attr_henc = self.text_encoder(attrs.to(torch.int64))
            attr_mu, attr_logvar = self._parameterize(attr_henc, attrs=True)
            mu = torch.cat((mu, attr_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, attr_logvar.unsqueeze(0)), dim=0)

        mu, logvar = self.expert(mu, logvar)

        z = self._mm_reparameterization(mu, logvar)

        image_recon = self.img_decoder(z)
        attr_recon = self.text_decoder(z)

        return {'recon_image': image_recon, 'recon_text': attr_recon, 'mu': mu, 'logvar': logvar}



class VaeBase(ReprLearner):
    """ Create Base class for VAE which inherits the architecture.
    """
    def __init__(self, img_encoder, img_decoder, **kwargs):
        super(VaeBase, self).__init__(**kwargs)
        self.img_encoder = img_encoder
        self.img_decoder = img_decoder
        #self.resnet = CustomizedResNet101()

    def _reparameterization(self, x):
        pass

    def forward(self, image):
        #image = self.resnet(image)
        h_enc = self.img_encoder(image)
        x = self._reparameterization(h_enc)
        x = self.img_decoder(x)

        return x


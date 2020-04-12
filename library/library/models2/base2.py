import pandas as pd
import numpy as np 
import pdb
import logging
import os

import torch
import torch.utils.data
from torch.nn import functional as F
from torch import nn, optim, Tensor
import torchvision.utils as vutils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReprLearner(nn.Module):
    """Representation Learner with PyTorch base functionality.
    """

    def __init__(self, **kwargs):
        super(ReprLearner, self).__init__(**kwargs)
        ## what kind of attributes are needed here!?
        self.loss_item = {}
    
    def _generate(self, x):
        """Given an input image x, returns the reconstructed image

        Args:

            model: 
            x: {Tensor} [B x C x H x W]
        Returns:
            {Tensor} [B x C H x W]
        """
        return self.forward(x.float())
        #return model(x.float())


    def _sample(self, num_samples):
        """Samples from the latent space and return the corresponding
        image space map.
        
        Args:
            num_samples {int}: number of samples
        Returns:
            {Tensor}
        """
        z = torch.randn(num_samples, 
                        self.latent_dim)

        #z = z.to(current_device)
        
        samples = self.img_decoder(z)
        #samples = model.img_decoder(z)
        return samples

    def _rep_evaluation(self, x):
        pass
    
    def any_other_function(self):
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
            experts,
            **kwargs):
        super(MMVaeBase, self).__init__(**kwargs)
        self.img_encoder = img_encoder
        self.img_decoder = img_decoder
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.experts = experts

    def _reparameterization(self, x):
        pass
        
    def forward(self, image=None, attrs=None):
        assert image is not None or attrs is not None
        if image is not None and attrs is not None:
            image_mu, image_logvar = self.img_encoder(image)
            attrs_mu, attrs_logvar = self.text_encoder(attrs)
            self.mu = torch.stack((image_mu, attrs_mu), dim=0)
            self.logvar = torch.stack((image_logvar, attr_logvar), dim=0)
        elif image is not None:
            mu, logvar = self.img_encoder(image)
            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        elif attrs is not None:
            mu, logvar = self.text_encoder(attrs)
            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        if self.img_encoder and self.text_encoder is not None:
            mu, logvar = self.experts(mu, logvar)
            
        z = self._reparameterization(mu, logvar)
            
        if self.img_encoder and self.text_encoder is not None:
            image_recon = self.img_decoder(z)
            attrs_recon = self.attrs_decoder(z)
            return image_recon, attrs_recon, mu, logvar
        elif image is not None:
            image_recon = self.img_decoder(z)
            return image_recon, mu, logvar
        elif attrs is not None:
            attrs_recon = self.attrs_decoder(z)
            return attrs_recon, mu, logvar

    def _sample_img_attrs(self):
        pass




class VaeBase(ReprLearner):
    """ Create Base class for VAE which inherits the architecture.
    """
    def __init__(self, img_encoder, img_decoder, **kwargs):
        super(VaeBase, self).__init__(**kwargs)
        self.img_encoder = img_encoder
        self.img_decoder = img_decoder

    def _reparameterization(self, x):
        pass

    def forward(self, x):
        h_enc = self.img_encoder(x)
        x = self._reparameterization(h_enc)
        x = self.img_decoder(x)
        return x

    def _sample_images(self,
                    val_gen,
                    epoch,
                    path, 
                    experiment_name):
        
        test_input, test_label = next(iter(val_gen))
        path = os.path.expanduser(path)
        storage_path = f"{path}{experiment_name}/"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        reconstruction = self._generate(test_input)
        vutils.save_image(reconstruction.data,
                        f"{storage_path}recon_{epoch}.png",
                        normalize=True,
                        nrow=12)
        try:
            #pdb.set_trace()
            samples = self._sample(num_samples=32)
            vutils.save_image(samples.data,
                            f"{storage_path}sample_{epoch}.png",
                           normalize=True,
                            nrow=12)
        except:
            pass
    
        del test_input, reconstruction
   
import pandas as pd
import numpy as np 
import pdb
import logging
import os
from library import architectures
from library.architectures import ConvEncoder, ConvDecoder


import torch
import torch.utils.data
from torch.autograd import Variable
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
#from torchvision.utils import save_image, vutils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReprLearner(nn.Module):
    """Representation Learner with PyTorch base functionality.
    """

    def __init__(self, **kwargs):
        super(ReprLearner, self).__init__(**kwargs)
        ## what kind of attributes are needed here!?
        self.loss_item = {}
        self.train_history = pd.DataFrame()
        self.train_val_history = pd.DataFrame()
        self.val_history = pd.DataFrame()

    def _train(self, model, train_gen, optimizer, optim_dict, lr_scheduler=False):
        """train function for a base NeuralNetwork in PyTorch.
        Args:
            n_epochs: {int} number of epochs
            batch_size: {int} size of batch 

            train_gen: {torch.data} iterable training dataset
            optim: {torch.optim} torch optimizer
            optim_dict: {dictionary} includes all relevant parameters for torch.optimizer
        """

        
        optimizer = optimizer(model.parameters(), **optim_dict)
        if lr_scheduler:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=0.95)
        model.float()
        model.train()

        for batch, (X, Y) in enumerate(train_gen):
            #pdb.set_trace()
            optimizer.zero_grad()
            x_recons = model(X.float())
            self.loss_item['recon_x'] = x_recons
            losses = self._loss_function(X.float(), **self.loss_item)
            #pdb.set_trace()
            loss = losses['loss']
            loss.backward()
            optimizer.step()


            if batch % 25 == 0:
                train_history = pd.DataFrame([[value.detach().numpy() for value in losses.values()]],
                                            columns=[key for key in losses.keys()])
            
                self.train_history = self.train_history.append(train_history).mean(axis=0)

                ## test lines: logger.info 
                logger.info('batch: {} | losses: {}'.format(batch, self.train_history))

        if lr_scheduler:
            scheduler.step()   


    #@classmethod
    def _validate(self, model, train_gen, val_gen, epoch):
        """validate function for a base NeuralNetwork in PyTorch.
        
        Args:
            n_epochs: {int} number of epochs
            val_gen: {torch.data} iterable validation dataset
        """
        model.eval()
        with torch.no_grad():

            for batch, (X, Y) in enumerate(train_gen):
                x_recons = model(X.float())
                self.loss_item['recon_x'] = x_recons
                losses = self._loss_function(X, **self.loss_item)

                train_val_history = pd.DataFrame([[value.detach().numpy() for value in losses.values()]],
                                                columns=[key for key in losses.keys()])
                
                self.train_val_history = self.train_val_history.append(train_val_history)

            for batch, (X, Y) in enumerate(val_gen):
                x_recons = model(X.float())
                self.loss_item['recon_x'] = x_recons
                losses = self._loss_function(X, **self.loss_item)

                val_history = pd.DataFrame([[value.detach().numpy() for value in losses.values()]],
                                            columns=[key for key in losses.keys()])
                
                self.val_history = self.val_history.append(val_history)

            train_summary = self.train_val_history.mean(axis=0)
            val_summary =  self.val_history.mean(axis=0)

            # looger statements:
            logger.info('epoch {} | train_losses: {} '.format(epoch, train_summary))
            logger.info('epoch {} | val_losses: {}'.format(epoch, val_summary))
    
    def _generate(self, model, x):
        """Given an input image x, returns the reconstructed image

        Args:
            model: 
            x: {Tensor} [B x C x H x W]
        Returns:
            {Tensor} [B x C H x W]
        """
        return model(x.float())

    def _sample(self, model, num_samples):
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
        
        samples = model.img_decoder(z)
        return samples

    def _rep_evaluation(self, x):
        pass


class MMVaeBase(ReprLearner):
    """Create a Base class for Multimodal VAE which inherits from ReprLearner.
    """     
    def __init__(self,
            img_encoder,
            img_decoder,
            text_encoder,
            text_decoder,
            experts):
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
        if img_encoder and text_encoder is not None:
            mu, logvar = self.experts(mu, logvar)
            
        z = self._reparameterization(mu, logvar)
            
        if img_encoder and text_encoder is not None:
            image_recon = self.img_decoder(z)
            attrs_recon = self.attrs_decoder(z)
            return image_recon, attrs_recon, mu, logvar
        elif image is not None:
            image_recon = self.img_decoder(z)
            return image_recon, mu, logvar
        elif attrs is not None:
            attrs_recon = self.attrs_decoder(z)
            return attrs_recon, mu, logvar





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
                    model,
                    val_gen,
                    path,
                    epoch,
                    experiment_name):
        test_input, test_label = next(iter(val_gen))
        #test_input = test_input.to(current_device)

        # check whether path already exists
        path = os.path.expanduser(path)
        storage_path = f"{path}{experiment_name}/"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        recons = model._generate(model, test_input)
        vutils.save_image(recons.data, 
                        f"{storage_path}recon_{epoch}.png",
                        normalize=True,
                        nrow=12
                        )
        
        try:
            samples = model._sample(model, num_samples=32)

            vutils.save_image(samples.data,
                            f"{storage_path}sample_{epoch}.png",
                            normalize=True,
                            nrow=12)
        except:
            pass
        
        del test_input, recons
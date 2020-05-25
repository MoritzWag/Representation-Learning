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



class VaeGaussian(VaeBase):
    """
    """
    def __init__(self, 
                **kwargs):
        super(VaeGaussian, self).__init__(
            **kwargs
        )
        self.latent_dim = self.img_encoder.latent_dim
        self.hidden_dim = self.img_encoder.hidden_dims
        self.mu = nn.Linear(self.hidden_dim[-1]*4, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dim[-1]*4, self.latent_dim)
        
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
        log_sigma = self.logvar(h_enc)

        std = torch.exp(0.5*log_sigma)
        eps = torch.randn_like(std)

        # store loss items
        self.loss_item['z_mean'] = mu
        self.loss_item['z_sigma'] = log_sigma

        return eps * std * mu


    def _loss_function(self, x: Tensor, recon_x: Tensor, z_mean: Tensor, z_sigma: Tensor):
        # latent loss: KL divergence
        def latent_loss(z_mean, z_sigma):
            return torch.mean(-0.5 * torch.sum(1 + z_sigma - z_mean ** 2 - z_sigma.exp(), dim=1), dim=0)
        
        # reconstruction loss 
        def reconstruction_loss(recon_x, x):
            return F.mse_loss(recon_x, x)
            #return F.binary_cross_entropy(recon_x, x)

        lat_loss = latent_loss(z_mean, z_sigma)
        recon_loss = reconstruction_loss(recon_x, x.float())

        kld_weight = 32 / 40000

        # should figure out how to properly weight the losses 
        loss = kld_weight * lat_loss + recon_loss 

        return {'loss': loss, 'lat_loss': lat_loss, 'recon_loss': recon_loss}


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



class BetaVae(VaeBase):
    """
    """
    def __init__(self,
                img_encoder=None,
                img_decoder=None,
                text_encoder=None,
                text_decoder=None,
                experts=None,
                latent_dim=None,
                categorical_dim=None,
                **kwargs):  
        super(BetaVae).__init__(
            **kwargs
        )

    
    def _reparameterization(self, h_enc):
        pass

    def _loss_function(self, x):
        pass



class VaeGumbel(VaeBase):
    """
    """
    def __init__(self,
            img_encoder=None, 
            img_decoder=None,
            text_encoder=None,
            text_decoder=None,
            experts=None,
            latent_dim=None,
            categorical_dim=None,
            temp=None):
        super(VaeGumbel).__init__(
            **kwargs
        )
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.temp = temp


    def _reparameterization(self, h_enc, eps=1e-7):
        """Gumpel Softmax reparameterization trick.
        
        Args:
            h_enc: {Tensor} 
        Return:
            Tensor
        """
        u = torch.rand_like(h_enc)
        g = - torch.log(- torch.log(u + eps) + eps)

        s = F.softmax((h_enc + g) / self.temp, dim=-1)
        s = s.view(-1, self.latent_dim*self.categorical_dim)

        return s 

    def _loss_function(self, x, x_recons):
        pass
        

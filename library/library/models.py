import pandas as pd
import numpy as np 
import pdb
from library import architectures
from library.architectures import ConvEncoder, ConvDecoder

import torch
import torch.utils.data
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class ReprLearner(nn.Module):
    """Neural Network with PyTorch base functionality.
    """

    def __init__(self):
        super(RepLearner, self).__init__()
        ## what kind of attributes are needed here!?
        ## self.history with pd.DataFrame => to then mflow tracking
        ## self.history = pd.DataFrame()

    def _train(self, model, train_gen, optim, optim_dict):
        """train function for a base NeuralNetwork in PyTorch.
        Args:
            n_epochs: {int} number of epochs
            batch_size: {int}
            train_gen: {torch.data} iterable training dataset
            optim: {torch.optim} torch optimizer
            optim_dict: {dictionary} includes all relevant parameters for torch.optimizer
        """
        
        #pdb.set_trace()
        model.float()
        optimizer = optim(model.parameters(), **optim_dict)
        #optimizer = optim(self.parameters(), **optim_dict)
        #optimizer = optim(list(self.model.parameters()), **optim_dict)
        model.train()
        for batch, (X, Y) in enumerate(train_gen):
            #pdb.set_trace()
            optimizer.zero_grad()
            #pdb.set_trace()
            x_recons = model(X.float())
            #loss = self._loss_function(**args)
            loss = self._loss_function(X, x_recons, z_mean=self.z_mean, z_sigma=self.z_sigma)
            #yhat = self.forward(X)
            #loss = self._loss_function(Y, yhat)
            loss.backward()
            optimizer.step()

            ## print statements for terminal!

    #@classmethod
    def _validate(self, model, val_gen):
        """validate function for a base NeuralNetwork in PyTorch.
        
        Args:
            n_epochs: {int} number of epochs
            val_gen: {torch.data} iterable validation dataset
        """
        reconstruction_loss = 0.0
        latent_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch, (X, Y) in enumerate(val_gen):
                x_recons = model(X)

        #self.history.append()

    #@classmethod
    def _predict(self):
        """predict function for a base Neural Network in PyTorch.
        
        Args:
        """
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for batch, (X, Y) in enumerate(test_loader):
                print(X)
                
                y_pred = self.model(X)

                ## estimate accuracy
    
    def save(self, sample_size):
        # save image
        with torch.no_grad():
            sample = torch.randn(sample_size, 20)
            sample = model(sample)
            save_image(sample.view(sample_size, 1, 28, 28),
                        'results/sample_{}.png'.format(blablab))
        


class VaeBase(ReprLearner):
    """ Create Base class for VAE which inherits the architecture.
    """
    def __init__(self, img_encoder,
            img_decoder,
            text_encoder,
            text_decoder,
            experts):
        super(VaeBase, self).__init__()
        self.img_encoder = img_encoder
        self.img_decoder = img_decoder
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.experts = experts

    #@classmethod
    def _reparameterization(self, x):
        pass

    #@classmethod
    def forward(self, x):
        #pdb.set_trace()
        h_enc = self.encoder(x)
        x = self._reparameterization(h_enc)
        x = self.decoder(x)
        return x
    
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
        
        if img_encoder and tex_encoder is not None:
            image_recon = self.img_decoder(z)
            attrs_recon = self.attrs_decoder(z)
            return image_recon, attrs_recon, mu, logvar
        elif image is not None:
            image_recon = self.img_decoder(z)
            return image_recon, mu, logvar
        elif attrs is not None:
            attrs_recon = self.attrs_decoder(z)
            return attrs_recon, mu, logvar



class VaeGaussian(VaeBase):
    """
    """
    def __init__(self, 
        	img_encoder=None,
            img_decoder=None,
            text_encoder=None, 
            text_decoder=None,
            experts=None,
            latent_dim=None):
        super(VaeGaussian, self).__init__(
            img_encoder, 
            img_decoder,
            text_encoder,
            text_decoder,
            experts)
        self.latent_dim = latent_dim
        #self.mu = nn.Linear(4096, 32)
        #self.log_sigma = nn.Linear(4096, 32)
        #self.fc3 = nn.Linear(32, 4096)
        #self.z_mean = None
        #self.z_sigma = None


    def _reparameterization(self, mu, logvar):
        ## Gaussian reparameterization: mu + epsilon*sigma
        #mu = self.mu(h_enc)
        #log_sigma = self.log_sigma(h_enc)
        #sigma = torch.exp(log_sigma)
        #std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        #self.z_mean = mu
        #self.z_sigma = sigma
        #return mu + sigma * Variable(std_z, requires_grad=False)

        # new implemented
        sigma = torch.exp(logvar)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        return mu + sigma * Variable(std_z, requires_grad=False)

    def _loss_function(self, x: torch.float32, recon_x: torch.float32, z_mean, z_sigma):
        # reconstruction loss
        def latent_loss(z_mean, z_sigma):
            return -0.5 * torch.mean(1 + z_sigma - z_mean.pow(2) - z_sigma.exp())

        def reconstruction_loss(recon_x, x):
            return F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

        return latent_loss(z_mean, z_sigma) + reconstruction_loss(recon_x, x)


class VaeBeta(VaeBase):
    """
    """
    def __init__(self, encoder, decoder):
        super(VaeBeta).__init__(encoder, decoder)

    
    def _reparameterization(self, h_enc):
        pass

    def _loss_function(self, x):
        pass



class VaeGumpel(VaeBase):
    """
    """
    def __init__(self,
            img_encoder=None, 
            img_decoder=None,
            text_encoder=None,
            text_decoder=None,
            experts=None,
            latent_dim,
            categorical_dim,
            temp):
        super(VaeGumpel).__init__(
            img_encoder,
            img_decoder, 
            text_encoder,
            text_decoder,
            experts)
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

    def _loss_function(self, x):
        pass
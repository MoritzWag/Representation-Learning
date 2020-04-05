import pandas as pd
import numpy as np 
import pdb
from library import architectures
from library.architectures import ConvEncoder, ConvDecoder

import torch
import torch.utils.data
from torch.autograd import Variable
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image



class ReprLearner(nn.Module):
    """Representation Learner with PyTorch base functionality.
    """

    def __init__(self, **kwargs):
        super(ReprLearner, self).__init__(**kwargs)
        ## what kind of attributes are needed here!?
        self.loss_item = {}
        self.train_losses = {}
        self.val_losses = {}
        self.test_losses = {}
        self.history = pd.DataFrame()

    def _train(self, model, train_gen, optim, optim_dict):
        """train function for a base NeuralNetwork in PyTorch.
        Args:
            n_epochs: {int} number of epochs
            batch_size: {int} size of batch 

            train_gen: {torch.data} iterable training dataset
            optim: {torch.optim} torch optimizer
            optim_dict: {dictionary} includes all relevant parameters for torch.optimizer
        """
        
        model.float()
        optimizer = optim(model.parameters(), **optim_dict)
        model.train()

        for batch, (X, Y) in enumerate(train_gen):
            
            optimizer.zero_grad()
            #pdb.set_trace()
            x_recons = model(X.float())
            self.loss_item['recon_x'] = x_recons
            loss = self._loss_function(X, **self.loss_item)
            # or: self.train_loss['train_loss'] += loss.item()
            self.train_losses['train_loss'] = loss.item()

            if batch % 5 == 0:
                print(x_recons[1, :, :, :])
            
            loss.backward()
            optimizer.step()

            #if epoch % blablab:
            #    # run through all keywords and items in losses dictionary
            #    print()

            ## print statements for terminal!
        #pdb.set_trace()
        self.history.append(self.train_losses, ignore_index=True)

        print('epoch {} | train_loss: {:0.5f} recon_loss: {:0.5f} lat_loss_ {:0.5f}'.format(**self.train_losses))   


    #@classmethod
    def _validate(self, model, val_gen, train_gen, val_steps):
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
                loss = self._loss_function(X, **self.loss_item)

                self.val_losses['val_loss'] = loss.item()
            
            self.history.append(self.val_losses, ignore_index=True)
            print()


    #@classmethod
    def _predict(self, model, test_gen):
        """predict function for a base Neural Network in PyTorch.
        
        Args:
        """
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for batch, (X, Y) in enumerate(test_gen):
                print(X)
                
                y_pred = model(X)
                ## estimate accuracy
    
    def _generate(self, model, N, K):
        pass

    def _sample(self, model, num_samples, latent_dim):
        """Samples from the latent space and return the corresponding
        image space map.
        
        Args:
            num_samples {int}: number of samples
        Returns:
            tensor
        """
        z = torch.randn(num_samples, latent_dim)
        samples = model.decoder(z)
        return samples

    def _save(self, model, sample_size):
        # save image
        with torch.no_grad():
            sample = torch.randn(sample_size, 20)
            sample = model(sample)
            save_image(sample.view(sample_size, 1, 28, 28),
                        'results/sample_{}.png'.format(blablab))


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





class VaeBase(ReprLearner):
    """ Create Base class for VAE which inherits the architecture.
    """
    def __init__(self, img_encoder, img_decoder, **kwargs):
        super(VaeBase, self).__init__(**kwargs)
        self.img_encoder = img_encoder
        self.img_decoder = img_decoder

    #@classmethod
    def _reparameterization(self, x):
        pass

    #@classmethod
    def forward(self, x):
        h_enc = self.img_encoder(x)
        x = self._reparameterization(h_enc)
        x = self.img_decoder(x)
        return x
    




class VaeGaussian(VaeBase):
    """
    """
    def __init__(self, 
        	#img_encoder=None,
            #img_decoder=None,
            text_encoder=None, 
            text_decoder=None,
            experts=None,
            latent_dim=None,
            loss_item=None,
            losses=None, 
            **kwargs):
        super(VaeGaussian, self).__init__(
            **kwargs
        )
        #super(VaeGaussian, self).__init__(
        #    img_encoder, 
        #    img_decoder,
        #    text_encoder,
        #    text_decoder,
        #    experts)
        #self.latent_dim = latent_dim
        self.mu = nn.Linear(4096, 32)
        self.logvar = nn.Linear(4096, 32)

    def _reparameterization(self, h_enc):
        ## Gaussian reparameterization: mu + epsilon*sigma
        mu = self.mu(h_enc)
        log_sigma = self.logvar(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        
        # store loss items
        self.loss_item['z_mean'] = mu
        self.loss_item['z_sigma'] = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)


    def _loss_function(self, x: Tensor, recon_x: Tensor, z_mean:Tensor, z_sigma:Tensor):
        # latent loss: KL divergence
        def latent_loss(z_mean, z_sigma):
            return -0.5 * torch.mean(1 + z_sigma - z_mean.pow(2) - z_sigma.exp())
        
        # reconstruction loss 
        def reconstruction_loss(recon_x, x):
            return torch.nn.functional.mse_loss(recon_x, x)

        lat_loss = latent_loss(z_mean, z_sigma)
        recon_loss = reconstruction_loss(recon_x, x.float())

        if self.training:
            self.train_losses['latent_loss'] = lat_loss.item()
            self.train_losses['reconstruction_loss'] = recon_loss.item()
        else:
            self.val_losses['latent_loss'] = lat_loss.item()
            self.val_losses['reconstruction_loss'] = recon_loss.item()

        return lat_loss + recon_loss


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
            latent_dim=None,
            categorical_dim=None,
            temp=None):
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
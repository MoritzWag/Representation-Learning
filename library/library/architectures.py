import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import pdb

class Encoder(nn.Module):
    """
    """
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, latent_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()


    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return x 
        


class Decoder(nn.Module):
    """
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.outpu_dim = ouput_dim
        self.fc1 = nn.Linear(latent_dim, )
        self.fc2 = nn.Linear()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return x


class ConvEncoder(nn.Module):
    """
    """
    def __init__(self, latent_dim=None):
        super(ConvEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.relu4 = nn.ReLU()
        self.flatten = Flatten()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.flatten(x)
        return x

class ConvDecoder(nn.Module):
    """
    """
    def __init__(self, latent_dim=None):
        super(ConvDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.unflatten = UnFlatten()
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=1)
        self.relu4 = nn.ReLU()
        self.fc1 = nn.Linear(32, 4096)

        self.mu = nn.Linear(4096, 32)
        self.logvar = nn.Linear(4096, 32)

    def forward(self, x):
        x = self.fc1(x)
        x = self.unflatten(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))

        return self.mu(x), self.logvar(x)


class Flatten(nn.Module):
    """
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

#class UnFlatten(nn.Module):
#    """
#    """
#    def __init__(self, size=256):
#        self.size = size
#
#    def forward(self, input):
#        return input.view(input.size(0), self.size, 4, 4)


class UnFlatten(nn.Module):
    """
    """
    def forward(self, input):
        return input.view(input.size(0), 256, 4, 4)


##################################
#
# Text Encoder and Decoder
# from: https://github.com/wenxuanliu/multimodal-vae/blob/6bbcd474c8736f7c802ce7c6574a17d0c9ebe404/celeba/model.py#L199
#
##################################

class AttributeEncoder(nn.Module):
    """
    """
    def __init__(self, n_latents):
        super(AttributeEncoder, self).__init__():
    
    def forward(self, z):
        pass



class AttributeDecoder(nn.Module):
    """
    """
    def __init__(self, n_latents):
        super(AttributeDecoder, self).__init__()
    
    def forward(self, z):
        pass



################################
#
# Product of Experts
#
################################

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    :param mu: M x D for M experts
    :param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        pd_mu = torch.sum(mu * var, dim=0) / torch.sum(var, dim=0)
        pd_var = 1 / torch.sum(1 / var, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

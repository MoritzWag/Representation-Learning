import torch
import torch.utils.data
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from torch import Tensor
import pdb

torch.set_default_dtype(torch.float64)

class ConvEncoder28x28(nn.Module):
    """
    """

    def __init__(self,
                in_channels: float,
                latent_dim: float,
                hidden_dims = None,
                **kwargs) -> None:
        super(ConvEncoder28x28, self).__init__()
            
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        #self.enc_output_dim = None
        
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        
        self.hidden_dims = hidden_dims

        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
    def forward(self, input: Tensor) -> Tensor:
        #pdb.set_trace()
        output = self.encoder(input)
        output = torch.flatten(output, start_dim=1)
        return output

    @property
    def enc_output_dim(self):
        dimensions = self.encoder(torch.ones(1, self.in_channels, 28, 28, requires_grad=False)).size()
        flattend_output_dimensions = dimensions[2] * dimensions[3]
        return flattend_output_dimensions


class ConvDecoder28x28(nn.Module):
    """
    """
    
    def __init__(self,
                in_channels: float,
                latent_dim: int,
                hidden_dims = None,
                categorical_dim = None,
                **kwargs) -> None:
        super(ConvDecoder28x28, self).__init__()
        self.in_channels = in_channels
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        modules = []
        
        if categorical_dim is None:
            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)
        else:
            self.categorical_dim = categorical_dim
            self.decoder_input = nn.Linear(latent_dim * categorical_dim, hidden_dims[-1]*4)
        
        hidden_dims.reverse()
        

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)

        #self.final_layer = nn.Sequential(
        #                        nn.ConvTranspose2d(hidden_dims[-1],
        #                                            hidden_dims[-1],
        #                                            kernel_size=3,
        #                                            stride=2,
        #                                            padding=1),
        #                        nn.BatchNorm2d(hidden_dims[-1]),
        #                        nn.LeakyReLU(),
        #                        nn.Conv2d(hidden_dims[-1], out_channels=1,
        #                                kernel_size=3, padding=1),
        #                       nn.Tanh())
        
        self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                    out_channels=self.in_channels,
                                                    kernel_size=2,
                                                    stride=2,
                                                    padding=2),
                                nn.BatchNorm2d(self.in_channels),
                                #nn.LeakyReLU(),
                                #nn.Conv2d(hidden_dims[-1], out_channels=1, 
                                #       kernel_size=3, padding=1),
                                nn.Tanh())


    def forward(self, input) -> Tensor:
        x = self.decoder_input(input)
        x = x.view(-1, 256, 2, 2)
        x = self.decoder(x)
        output = self.final_layer(x)
        return output



class ConvEncoder64x64(nn.Module):
    """
    """

    def __init__(self,
                in_channels: int,
                latent_dim: int,
                hidden_dims = None,
                **kwargs) -> None:
        super(ConvEncoder64x64, self).__init__()
            
        
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.hidden_dims = hidden_dims

        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)

    def forward(self, input: Tensor) -> Tensor:
        output = self.encoder(input)
        output = torch.flatten(output, start_dim=1)
        return output




class ConvDecoder64x64(nn.Module):
    """
    """
    
    def __init__(self,
                latent_dim: int,
                hidden_dims = None,
                **kwargs) -> None:
        super(ConvDecoder64x64, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)

        hidden_dims.reverse()
        

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                    hidden_dims[-1],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    output_padding=1),
                                nn.BatchNorm2d(hidden_dims[-1]),
                                nn.LeakyReLU(),
                                nn.Conv2d(hidden_dims[-1], out_channels=3,
                                        kernel_size=3, padding=1),
                                nn.Tanh())
        
        self.final_layer224x224 = nn.Sequential(
                                    nn.ConvTranspose2d(3,
                                                    3,
                                                    kernel_size=3,
                                                    stride=3, 
                                                    padding=1),
                                    nn.BatchNorm2d(3),
                                    #nn.LeakyReLU(),
                                    #nn.ConvTranspose2d(3,
                                    #                3,
                                    #                kernel_size=3,
                                    #                stride=3,
                                    #                padding=1),
                                    #nn.BatchNorm2d(3),
                                    #nn.Tanh())
                                    nn.Sigmoid())

    def forward(self, input) -> Tensor:
        #pdb.set_trace()
        x = self.decoder_input(input)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        output = self.final_layer(x)
        output = self.final_layer224x224(output)
        return output



class ConvEncoder224x224(nn.Module):
    """
    """

    def __init__(self, 
                in_channels: int,
                latent_dim: int,
                hidden_dims = None,
                **kwargs) -> None:
        super(ConvEncoder224x224, self).__init__()

        self.latent_dim = latent_dim
        
        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64, 128, 256, 512]
        
        self.hidden_dims = hidden_dims

        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim 
        
        self.encoder = nn.Sequential(*modules)
    
    def forward(self, input: Tensor) -> Tensor:
        output = self.encoder(input)
        output = torch.flatten(output, start_dim=1)
        return output
    
    @property
    def enc_output_dim(self):
        dimensions = self.encoder(torch.ones(1, self.in_channels, 224, 224, requires_grad=False)).size()
        flattend_output_dimensions = dimensions[2] * dimensions[3]
        return flattend_output_dimensions





class ConvDecoder224x224(nn.Module):
    """
    """
    def __init__(self,
                latent_dim: int,
                hidden_dims = None,
                **kwargs) -> None:
        super(ConvDecoder224x224, self).__init__()

        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64, 128, 256, 512]
        
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        # 128x128
        self.decoder = nn.Sequential(*modules)

        self.final_layer1 = nn.Sequential(
                                    nn.ConvTranspose2d(hidden_dims[-1],
                                                        hidden_dims[-1],
                                                        kernel_size=3,
                                                        stride=2,
                                                        padding=1),
                                    nn.BatchNorm2d(hidden_dims[-1]),
                                    nn.LeakyReLU())

        self.final_layer2 = nn.Sequential(
                                nn.Conv2d(hidden_dims[-1], out_channels=3,
                                        kernel_size=32),
                                nn.Tanh())

    def forward(self, input: Tensor) -> Tensor:
        x = self.decoder_input(input)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        output = self.final_layer1(x)
        output = self.final_layer2(output)
        return output
























##################################
#
# Text Encoder and Decoder
# from: https://github.com/wenxuanliu/multimodal-vae/blob/6bbcd474c8736f7c802ce7c6574a17d0c9ebe404/celeba/model.py#L199
#
##################################

class AttributeEncoder(nn.Module):
    """
    """
    def __init__(self, num_attr):
        super(AttributeEncoder, self).__init__()
        self.num_attr = num_attr
        self.net = nn.Sequential(
                    nn.Embedding(10, 50),
                    nn.BatchNorm1d(50),
                    nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.net(x)
        return x



class AttributeDecoder(nn.Module):
    """
    """
    def __init__(self, num_attr):
        super(AttributeDecoder, self).__init__()
        self.num_attr = num_attr
        self.net = nn.Sequential(
                    nn.Linear(num_attr, 10),
                    nn.BatchNorm1d(10),
                    nn.ReLU(),
                    nn.Linear(10, 10)

        )
    
    def forward(self, z):
        z = self.net(z)
        return F.log_softmax(z)


#########################################
#
#
# Mulit-Attributes Encoder/Decoder
#
##########################################


class MultiAttrEncoder(nn.Module):
    """
    """
    def __init__(self,
                list_num_categories,
                list_num_attributes):
        super(MultiAttrEncoder, self).__init__()
        self.list_num_categories = list_num_categories
        self.list_num_attributes = list_num_attributes
        self.net = nn.Sequential()

    def forward(self):
        pass


class MultiAttrDecoder(nn.Module):
    """
    """
    def __init__(self,
                list_num_categories,
                list_num_attributes):
        super(MultiAttrDecoder, self).__init__()
        self.list_num_categories = list_num_categories
        self.list_num_attributes = list_num_attributes
        self.net = nn.Sequential()

    
    def forward(self):
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


def prior_experts(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical G
    Gaussian: N(0, 1).

    Args:
        size: {integer} dimensionality of Gaussian
        use_cuda: {boolena} cast CUDA on variables
    """

    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar



####################################
#
# Mixture of Experts
# https://github.com/iffsid/mmvae/blob/d988793447453565122b6bab1fdf1df18d2f74e9/src/models/mmvae.py#L65
#
#####################################








##############################################################
#
#
# pre-trained ResNet 101 on ImageNet (adjusted)
#
#
###############################################################


class CustomizedResNet101(nn.Module):
    """
    """
    def __init__(self, pretrained=True):
        super(CustomizedResNet101, self).__init__()

        if pretrained:
            resnet101 = models.resnet101(pretrained=True)
            modules = list(resnet101.children())[:-2]
            self.resnet101 = nn.Sequential(*modules)
            for p in self.resnet101.parameters():
                p.requires_grad = False

        self.linear = nn.Linear(2048*7*7, 12288)

    def forward(self, x):
        output = self.resnet101(x)
        output = torch.flatten(output, 1)
        output = self.linear(output)
        output = output.view(32, 3, 64, 64)
        return output


    
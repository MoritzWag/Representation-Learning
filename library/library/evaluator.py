import numpy as np 
import pandas as pd 
import os
import torch 
import math 
from torch import nn 
import pdb 
from tqdm import tqdm, trange

from library.eval_helpers import log_density_gaussian


class Evaluator(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)

    def total_correlation(self):
        pass

    def total_wasserstein_correlation(self):
        pass

    def misjed(self):
        """Calculates the  Mutual Information Sums Joint Entropy Difference

        Args:

        Returns:
            misjed: {torch.Tensor} a matrix of size (latent_dim, latent_dim)
        """
        pass

    def windin(self):
        pass

    def entropy(self):
        pass

    def mutual_information(variables):

        if len(variables) < 2:
            raise AttributeError(
                "Mutual information must involve at least 2 variables")
        all_vars = np.hstack(variables)

        #return (sum([entropy(X, type=['marginal'] for X in variables)]
        #        - entropy(all_vars, type=['joint']))

    def get_params(self, data):
        mus = []
        logvars = []
        zs = []
        for batch, (image, attribute) in enumerate(data):
            mu, logvar, _, z = self._embed(image)
            mus.append(mu)
            logvars.append(logvar)
            zs.append(z)

        mus = np.stack(mus)
        mus = mus.reshape(mus.shape[0]*mus.shape[1], -1)

        logvars = np.stack(logvars)
        logvars = logvars.reshape(logvars.shape[0]*logvars.shape[1], -1)

        zs = np.stack(zs)
        zs = zs.reshape(zs.shape[0]*zs.shape[1], -1)

        mus = torch.from_numpy(mus)
        logvars = torch.from_numpy(logvars)
        zs = torch.from_numpy(zs)

        return [mus, logvars, zs]

    def _estimate_H_z(self, params, n_samples, type=['single', 'average']):
        """
        """
        mus = params[0]
        logvars = params[1]
        zs = params[2]

        len_dataset, latent_dim = zs.size(0), zs.size(1)

        samples_x = torch.randperm(len_dataset)[:n_samples]
        samples_zCx = zs.index_select(0, samples_x).view(latent_dim, n_samples)

        mini_batch_size = 10
        samples_zCx = samples_zCx.expand(len_dataset, latent_dim, n_samples)
        mean = mus.unsqueeze(-1).expand(len_dataset, latent_dim, n_samples)
        log_var = logvars.unsqueeze(-1).expand(len_dataset, latent_dim, n_samples)
        log_N = math.log(len_dataset)

        pdb.set_trace()

        H_z = torch.zeros(latent_dim)
        with trange(n_samples, leave=False) as t:
            for k in range(0, n_samples, mini_batch_size):
                idx = slice(k, k + mini_batch_size)
                log_q_zCx = log_density_gaussian(samples_zCx[..., idx],
                                                mean[..., idx],
                                                log_var[..., idx])
                
                log_q_z = -log_N + torch.logsumexp(log_q_zCx, dim=0, keepdim=False)

                H_z += (-log_q_z).sum(1)

        H_z /= n_samples

        return H_z



    def _estimate_H_zCz(self, params, n_samples):
        """
        """
        mus = params[0]
        logvars = params[1]
        zs = params[2]

        len_dataset, latent_dim = zs.size(0), zs.size(1)
        H_zCz = torch.zeros(latent_dim, latent_dim)
        
        

    def estimate_H_zJz(self):
        pass

    def _estimate_latent_entropies(self, data, n_samples, type=['marginal', 'joint']):
        """
        https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/evaluate.py#L196
        """
        mu = []
        logvar = []
        embedding = []
        for batch, (image, attribute) in enumerate(data):
            mu, logvar, emebdding = self._embed(image)
            mu.append(mu)
            logvar.append(logvar)
            embedding.append(emebdding)

        if type == 'marginal':
            pass 


        if type == 'joint':
            pass
        
        # a vector of size (1, latent_dim)
        H_z = torch.zeros(latent_dim, device=device)



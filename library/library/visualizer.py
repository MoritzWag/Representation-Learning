import numpy as np 
import pandas as pd 
import os
import torch 
from torch import nn
from scipy import stats 
import pdb

import torchvision.utils as vutils
from library.viz_helpers import sort_list_by_other

class Visualizer(nn.Module):

    def __init__(self, **kwargs):
        super(Visualizer, self).__init__(**kwargs)


    def _get_traversal_range(self, max_traversal=10, mean=0, std=1):
        """Return the corresponding traversal range in absolute terms.

        Args:
            max_traversal: {int}
            mean: {float}
            std: {float} 
        """
        
        if max_traversal < 0.5:
            max_traversal = (1 - 2 * max_traversal) / 2
            max_traversal = stats.norm.ppf(max_traversal, loc=mean, scale=std)
        
        return (-1 * max_traversal, max_traversal)
    
    def _traverse_line(self, idx, n_samples, embedding=None):
        """
        Args:
            embedding: {torch} represents the 
        """

        if embedding is None:
            samples = torch.zeros(n_samples, self.latent_dim)
            traversals = torch.linspace(*self._get_traversal_range(), steps=n_samples)
        else:
            if embedding.size(0) > 1:
                #raise ValueError('Every value should be sampled from the same posterior, but {} datapoints given'.format(data.size(0)))
                embedding = embedding[0,:]
            with torch.no_grad():
                # here, distinction between VaeBase and MMVaeBase must be made!
                mu, logvar, embed = self._embedding(embedding)
                mu = mu.unsqueeze(0)
                logvar = logvar.unsqueeze(0)
                samples = self._reparameterization(embed)
                samples = samples.repeat(n_samples, 1)
                post_mean_idx = mu[0, idx]
                post_std_idx = torch.exp(logvar / 2)[0, idx]
            
            traversals = torch.linspace(*self._get_traversal_range(mean=post_mean_idx,
                                                                std=post_std_idx),
                                        steps=n_samples)
        
        for i in range(n_samples):
            samples[i, idx] = traversals[i]
        
        return samples

    def traversals(self, 
                epoch,
                experiment_name,
                path,
                embedding=None,
                is_reorder_latents=False,
                n_per_latent=8,
                n_latents=None):

        n_latents = n_latents if n_latents is not None else self.latent_dim
        latent_samples = [self._traverse_line(dim, n_per_latent, embedding=embedding) for
                            dim in range(self.latent_dim)]
        decoded_traversals = self.img_decoder(torch.cat(latent_samples, dim=0))

        if is_reorder_latents:
            n_images, *other_shape = decoded_traversals.size()
            n_rows = n_images // n_per_latent
            decoded_traversals = decoded_traversals.reshape(n_rows, n_per_latent, *other_shape)
            decoded_traversals = sort_list_by_other(decoded_traversals, self.losses)
            decoded_traversals = torch.stack(decoded_traversals, dim=0)
            decoded_traversals = decoded_traversals.reshape(n_images, *other_shape)

        path = os.path.expanduser(path)
        storage_path = f"{path}{experiment_name}/"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        vutils.save_image(decoded_traversals.data,
                        f"{storage_path}traversal_{epoch}.png",
                        normalize=True,
                        nrow=10)

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
            samples = self._sample(num_samples=32)
            vutils.save_image(samples.data,
                        f"{storage_path}recon_{epoch}.png",
                        normalize=True,
                        nrow=12)
        except:
            pass

        del test_input, reconstruction
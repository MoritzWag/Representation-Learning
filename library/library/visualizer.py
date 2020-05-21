import numpy as np 
import pandas as pd 
import os
import torch 
from torch import nn
from scipy import stats 
import pdb

import torchvision.utils as vutils
from library.viz_helpers import sort_list_by_other

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

torch.set_default_dtype(torch.float64)

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
    
    def _traverse_line(self, idx, n_samples, data=None):
        """
        Args:
            embedding: {torch} represents the 
        """

        if data is None:
            samples = torch.zeros(n_samples, self.latent_dim)
            traversals = torch.linspace(*self._get_traversal_range(), steps=n_samples)
        else:
            with torch.no_grad():
                image, attribute = next(iter(data))
                if torch.cuda.is_available():
                    image, attribute = image.cuda(), attribute.cuda()
                mu, logvar, embedding = self._embed(image.float())

                if embedding.size(0) > 1:
                #raise ValueError('Every value should be sampled from the same posterior, but {} datapoints given'.format(data.size(0)))
                    embedding = embedding[0, :]
                    mu = mu[0, :]
                    logvar = logvar[0, :]
                mu = mu.unsqueeze(0)
                logvar = logvar.unsqueeze(0)
                samples = self._reparameterization(h_enc=embedding)
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
                data=None,
                is_reorder_latents=False,
                n_per_latent=8,
                n_latents=None):

        #if experiment_name is not 
        n_latents = n_latents if n_latents is not None else self.latent_dim
        latent_samples = [self._traverse_line(dim, n_per_latent, data=data) for
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
                        nrow=n_per_latent)

        
        # CatVae traversals:
        # Example: latent_dim = 5, cat_dim = 5
        # 1.) image, attribute next(iter(data))
        # 2.) embedding, z = self._emebd(img.float())
        # 3.) dim(z) = 32, 5*5 => (32, 25)
        # 4.) z = z[0, :] => dim(z) = (1, 25)
        # 4.) for lat_dim in self.latent_dim:
        #        for cat_dim in self.cat_dim:
        #            z[lat_dim*cat_dim] = 1 
        #            z




    def _sample_images(self,
                    val_gen,
                    epoch,
                    path,
                    experiment_name):

        test_input, test_label = next(iter(val_gen))

        if torch.cuda.is_available():
            test_input, test_label = test_input.cuda(), test_label.cuda()

        path = os.path.expanduser(path)
        storage_path = f"{path}{experiment_name}/"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        reconstruction = self._generate(test_input)
        vutils.save_image(reconstruction.data,
                        f"{storage_path}recon_{epoch}.png",
                        normalize=True,
                        nrow=12)
        vutils.save_image(test_input.data,
                        f"{storage_path}real_{epoch}.png",
                        normalize=True,
                        nrow=12)
        try:
            samples = self._sample(num_samples=32)
            vutils.save_image(samples.data,
                        f"{storage_path}sample_{epoch}.png",
                        normalize=True,
                        nrow=12)
        except:
            print("could not sample images!")

        del test_input, reconstruction

    def _cluster(self,
                data,
                path, 
                epoch,
                experiment_name,
                num_batches=10):
        """Clustering algorithm with t-SNE visualization capability
        Args:
            feature_list {}: 
            path {str}:
            experiment_name {}:
        """
        
        indices = np.random.choice(a=len(data),
                                    size=int(num_batches),
                                    replace=False)
        features_extracted = []
        features_labels = []
        for batch, (image, attribute) in enumerate(data):
            if batch in indices:
                if torch.cuda.is_available():
                    image = image.cuda()
                h_enc = self.img_encoder(image.float())
                z = self._reparameterization(h_enc)
                z = z.cpu().detach().numpy()
                features_extracted.append(z)
                features_labels.append(attribute)
            else:
                pass
        features_extracted = np.vstack(features_extracted)
        features_labels = np.concatenate(features_labels)

        ## t-SNE:
        tsne_results = TSNE(n_components=2, verbose=1, metric='euclidean',
                            perplexity=50, n_iter=1000, learning_rate=200).fit_transform(features_extracted)

        ## plot t-SNE results:
        plt.close()
        colormap = plt.cm.get_cmap('coolwarm')
        scatter_plot = plt.scatter(tsne_results[:, 0], tsne_results[:,1], 
                                c=features_labels, cmap=colormap)
        plt.colorbar(scatter_plot)
 
        path = os.path.expanduser(path)
        storage_path = f"{path}{experiment_name}/"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        img_storage_path =  f"{storage_path}/cluster_{epoch}"
        plt.tight_layout()
        plt.savefig(img_storage_path)
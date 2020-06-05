import numpy as np 
import pandas as pd 
import os
import torch 
from torch import nn
from scipy import stats 
import pdb
import cv2
from tqdm import tqdm

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
    
    def _traverse_line(self, idx, n_samples, model_type, data=None):
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
                
                if model_type != 'CatVae':

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
                else:
                    code = self._embed(image.float())
                    code = code[0,:].view(self.latent_dim, self.categorical_dim)
                    total_dim = self.latent_dim * self.categorical_dim
                    traversal = torch.rand(
                        total_dim,
                        self.latent_dim,
                        self.categorical_dim
                    )
                    pdb.set_trace()
                    total = 0
                    for latent in range(self.latent_dim):
                        for cat in range(self.categorical_dim):
                            
                            traversal[total] = code
                            one_hot = torch.zeros(self.categorical_dim)
                            one_hot[cat] = 1
                            traversal[total, latent, :] = one_hot 

                            total =+ 1
            
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
        
        #if self.__class__.__name__ != 'CatVae':
        n_latents = n_latents if n_latents is not None else self.latent_dim
        latent_samples = [self._traverse_line(dim, n_per_latent, data=data, model_type=self.__class__.__name__) for
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

    def get_coordinates(self, image, dimred_x, dimred_y, global_size):
        # changed from https://www.learnopencv.com/t-sne-for-feature-visualization/
        # Get height and width of image
        height, width = image.shape
        
        # compute the image center coordinates for dimensionality reduction plot
        center_x = int(global_size * dimred_x)

	    # to have the same here, we need to mirror the y coordinate
        center_y = int(global_size * (1 - dimred_y))

        # Compute edge coordinates
        topleft_x = center_x - int(width / 2)
        topleft_y = center_y - int(height / 2)

        bottomright_x = center_x + int(width / 2)
        bottomright_y = center_y + int(height / 2)

        if topleft_x < 0:
            bottomright_x = bottomright_x + abs(topleft_x)
            topleft_x = topleft_x + abs(topleft_x)
        
        if topleft_y < 0:
            bottomright_y = bottomright_y + abs(topleft_y)
            topleft_y = topleft_y + abs(topleft_y)

        if bottomright_x > global_size:
            topleft_x = topleft_x - (bottomright_x - global_size)
            bottomright_x = bottomright_x - (bottomright_x - global_size)

        if bottomright_y > global_size:
            topleft_y = topleft_y - (bottomright_y - global_size)
            bottomright_y = bottomright_y - (bottomright_y - global_size)
        
        return topleft_x, topleft_y, bottomright_x, bottomright_y

    def reshape_image(self, img, scaling):
        """
        Input: NumpyArray {[H,W,C]}
        Output: Resized numpy array {[H,W,C]}
        """
        # Undo scaling
        img = img * 255

        width = int(img.shape[1] * (scaling / 100))
        height = int(img.shape[0] * (scaling / 100))
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

        return resized

    def _cluster(self,
                data,
                path, 
                epoch,
                experiment_name,
                num_batches=10,
                plot_images = True,
                plot_size = 1000):
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
        images_ = None

        for batch, (image, attribute) in enumerate(data):
            if batch in indices:
                if torch.cuda.is_available():
                    image = image.cuda()
                h_enc = self.img_encoder(image.float())
                z = self._reparameterization(h_enc)
                z = z.cpu().detach().numpy()
                features_extracted.append(z)
                features_labels.append(attribute)
                #images_.append(image)
                if images_ is None:
                    images_ = image.numpy().transpose(0, 2, 3, 1)
                image = image.numpy().transpose(0, 2, 3, 1)
                images_ = np.append(images_, image, 0)
            else:
                pass
        features_extracted = np.vstack(features_extracted)
        features_labels = np.concatenate(features_labels)

        ## t-SNE:
        tsne_results = TSNE(n_components=2, verbose=1, metric='euclidean',
                            perplexity=50, n_iter=1000, learning_rate=200).fit_transform(features_extracted)

        tx, ty = tsne_results[:,0], tsne_results[:,1]
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

        path = os.path.expanduser(path)
        storage_path = f"{path}{experiment_name}/"
        if not os.path.exists(storage_path):
                os.makedirs(storage_path)

        if plot_images: 
            # Create blank canvas to be filled with images
            if  self.img_encoder.in_channels == 1:
                tsne_plot = 255 * np.ones((plot_size, plot_size), np.uint8)
            else:
                tsne_plot = 255 * np.ones((plot_size, plot_size, self.img_encoder.in_channels), np.uint8)

            # Fill the blank plot with the coordinates of the images according to tSNE
            for img, label, x, y in tqdm(zip(images_, features_labels, tx, ty),
            desc='Plotting t-SNE with images',
            total=len(images_)):

                img = self.reshape_image(img, 100)
                tl_x, tl_y, br_x, br_y = self.get_coordinates(img, x, y, plot_size)

                # draw a rectangle with a color corresponding to the image class
	            #image = draw_rectangle_by_class(img, label)
                #tsne_plot[tl_y:br_y, tl_x:br_x, :] = img
                tsne_plot[tl_y:br_y, tl_x:br_x] = img 
            
            img_storage_path =  f"{storage_path}/cluster_{epoch}.png"
            cv2.imwrite(img_storage_path, tsne_plot)

        else:
            ## plot t-SNE results:
            plt.close()
            colormap = plt.cm.get_cmap('coolwarm')
            scatter_plot = plt.scatter(tsne_results[:, 0], tsne_results[:,1], 
                                    c=features_labels, cmap=colormap)
            plt.colorbar(scatter_plot)

            img_storage_path =  f"{storage_path}/cluster_{epoch}"
            plt.tight_layout()
            plt.savefig(img_storage_path)




        
    



import numpy as np
import pandas as pd
import seaborn as sns
import os
import torch
from torch import nn
from scipy import stats
import pdb
import cv2
import copy
from tqdm import tqdm

import torchvision.utils as vutils
from library.viz_helpers import sort_list_by_other

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap  

torch.set_default_dtype(torch.float64)


class Visualizer(nn.Module):

    def __init__(self, **kwargs):
        super(Visualizer, self).__init__(**kwargs)

    def _get_traversals(self, model_type, data=None, normal_traversals = False):
        """
        Args:
            embedding: {torch} represents the
        """

        with torch.no_grad():
            
            if model_type == 'GaussmixVae':
            
                probabilities = torch.tensor(
                    [0.01, 0.05, 0.15, 0.25, 0.45, 0.5, 0.55, 0.75, 0.85, 0.95, 0.99]
                )

                num_latent_trav = probabilities.size()[0]

                if normal_traversals == False:
                    probs = copy.deepcopy(self.store_probs.cpu().numpy())
                    z = copy.deepcopy(self.store_individual_z.view(-1, self.categorical_dim * self.latent_dim).transpose(0,1).cpu().numpy())

                    quantiles = np.empty((self.latent_dim*self.categorical_dim, num_latent_trav))
                    for i in range(self.latent_dim):
                        quantiles[i, :] = [np.quantile(z[i,:], prob) for prob in probabilities] 
                    
                    quantiles = torch.tensor(quantiles)

                else:
                    mu_hat = self.mu_hat.view(
                        self.latent_dim * self.categorical_dim)
                    sigma_hat = self.sigma_hat.view(
                        self.latent_dim * self.categorical_dim)
                    normal_dist = [
                        torch.distributions.normal.Normal(
                            x, y) for x, y in zip(
                            mu_hat, sigma_hat)]

                    quantiles = torch.stack([normal_dist[x].icdf(
                        probabilities) for x in range(len(normal_dist))])
                        
                quantiles = quantiles.view(
                        self.categorical_dim,
                        self.latent_dim,
                        probabilities.size()[-1]
                )

                traversal = torch.rand(
                    self.categorical_dim * self.latent_dim *
                    probabilities.size()[-1],
                    self.latent_dim
                )

                total = 0
                with tqdm(total = self.latent_dim * len(probabilities) * self.categorical_dim,
                desc='Buildung traversal plots') as pbar:
                    for cat in range(self.categorical_dim):
                        for lat in range(self.latent_dim):
                            for prob in range(len(probabilities)):
                                mu = copy.deepcopy(self.mu_hat[cat])
                                mu[lat] = quantiles[cat, lat, prob]
                                traversal[total, :] = mu
                                total += 1
                                pbar.update(1)

                if torch.cuda.is_available():
                    traversal = traversal.cuda()

                traversal = traversal.view(
                    self.categorical_dim,
                    self.latent_dim * probabilities.size()[-1],
                    self.latent_dim
                )

            elif model_type == 'CatVae':

                num_latent_trav = self.categorical_dim

                total_dim = self.latent_dim * self.categorical_dim
                traversal = torch.rand(
                    total_dim,
                    self.latent_dim,
                    self.categorical_dim
                )

                const_one_hot = torch.tensor(
                    np.concatenate(
                        (np.array([1]), np.repeat(0, self.categorical_dim - 1))
                    )
                ).float()

                code = torch.cat(
                    self.latent_dim * [const_one_hot]
                ).view(self.latent_dim,self.categorical_dim)

                total = 0
                for latent in range(self.latent_dim):
                    for cat in range(self.categorical_dim):

                        traversal[total] = code
                        one_hot = torch.zeros(self.categorical_dim)
                        one_hot[cat] = 1
                        traversal[total, latent, :] = one_hot

                        total += 1

                    traversal = traversal.view(-1, total_dim).float()

                if torch.cuda.is_available():
                    traversal = traversal.cuda()

            else:
                # Get the probabilities for the quantile function
                probabilities = torch.tensor(
                [0.001, 0.01, 0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99, 0.999]
                )

                num_latent_trav = probabilities.size()[0]

                if normal_traversals == False:
                    
                    # Compute emprical quantiels according to "probabilities"
                    z = copy.deepcopy(self.store_z.transpose(0,1).cpu().numpy())

                    quantiles = np.empty((self.latent_dim, num_latent_trav))
                    for i in range(self.latent_dim):
                        quantiles[i, :] = [np.quantile(z[i,:], prob) for prob in probabilities] 
                    
                    quantiles = torch.tensor(quantiles)

                else:
                    # Get the estimates for the means and the variance for each
                    # latent to parameterize a normal distribution (1 Normal Dist 
                    # per latent)
                    mu_hat = self.mu_hat
                    sigma_hat = self.sigma_hat
                    normal_dist = [
                        torch.distributions.normal.Normal(
                            x, y) for x, y in zip(
                            mu_hat, sigma_hat
                        )
                    ]

                    # Apply quantile function with the specified proabilities to 
                    # each normal distribution and change the shape
                    quantiles = torch.stack([
                        normal_dist[x].icdf(
                        probabilities) for x in range(len(normal_dist))
                        ]
                    )
                    
                quantiles = quantiles.view(
                    self.latent_dim, probabilities.size()[-1]
                )

                # Create dummy traversal tensor to be filled up in subsequent
                # loop. The N-Dimension will be N = Latent_Dim * Probabilities.
                # This is the total number of plots which shall be plotted
                traversal = torch.rand(
                    self.latent_dim * probabilities.size()[-1],
                    self.latent_dim
                )
                total = 0
                median = copy.deepcopy(quantiles[:, 6])
                with tqdm(total = self.latent_dim * len(probabilities),
                desc='Buildung traversal plots') as pbar:
                    for lat in range(self.latent_dim):
                        for prob in range(len(probabilities)):
                            med = copy.deepcopy(median)
                            med[lat] = quantiles[lat, prob]
                            traversal[total, :] = med
                            total += 1
                            pbar.update(1)

                if torch.cuda.is_available():
                    traversal = traversal.cuda()

            return traversal.float(), num_latent_trav

    def traversals(self,
                   epoch,
                   run_name,
                   path,
                   data=None):

        self.epoch = epoch

        carry_on = (epoch <= 10) or ((epoch % 20) == 0)
        if not carry_on:
            return
        
        # latent_traversals_normal, n_per_latent = self._get_traversals(
        #     model_type = self.__class__.__name__,
        #     data = data,
        #     normal_traversals = True
        # )

        latent_traversals_empirical, n_per_latent = self._get_traversals(
            model_type = self.__class__.__name__,
            data = data,
            normal_traversals = False
        )

        path = os.path.expanduser(path)
        storage_path = f"{path}{run_name}/"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        if self.__class__.__name__ == 'GaussmixVae':
            for categories in range(self.categorical_dim):

                # decoded_traversals_normal = self.img_decoder(
                #     latent_traversals_normal[categories]
                # )

                decoded_traversals_empirical = self.img_decoder(
                    latent_traversals_empirical[categories]
                )

                # vutils.save_image(
                #     decoded_traversals_normal.data,
                #     f"{storage_path}traversal_norm_{epoch}_cat{categories+1}.png",
                #     normalize=True,
                #     nrow=n_per_latent
                # )

                vutils.save_image(
                    decoded_traversals_empirical.data,
                    f"{storage_path}traversal_emp_{epoch}_cat{categories+1}.png",
                    normalize=True,
                    nrow=n_per_latent
                )
        else:
            # decoded_traversals_normal = self.img_decoder(
            #     latent_traversals_normal
            # )

            decoded_traversals_empirical = self.img_decoder(
                latent_traversals_empirical
            )

            # vutils.save_image(
            #     decoded_traversals_normal.data,
            #     f"{storage_path}traversal_norm_{epoch}.png",
            #     normalize=True,
            #     nrow=n_per_latent
            # )

            vutils.save_image(
                decoded_traversals_empirical.data,
                f"{storage_path}traversal_emp_{epoch}.png",
                normalize=True,
                nrow=n_per_latent
            )

    def _sample_images(self,
                       image,
                       epoch,
                       path,
                       run_name):

        carry_on = (epoch <= 10) or ((epoch % 20) == 0)
        if not carry_on:
            return

        path = os.path.expanduser(path)
        storage_path = f"{path}{run_name}/"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        reconstruction = self._generate(image[:32])

        recon_real = torch.cat(
            (image[:32].cuda(), reconstruction.type(
                torch.DoubleTensor).cuda()), 0)
        vutils.save_image(recon_real.data,
                          f"{storage_path}real_recon{epoch}.png",
                          normalize=True,
                          nrow=8)

        try:
            samples = self._sample(num_samples=32)
            vutils.save_image(samples.data,
                              f"{storage_path}sample_{epoch}.png",
                              normalize=True,
                              nrow=8)
        except BaseException:
            print("could not sample images!")

        del reconstruction

    def _cluster(self,
                 image,
                 attribute,
                 path,
                 epoch,
                 run_name,
                 method='umap',
                 num_samples=320,
                 plot_size=1000):
        """Clustering algorithm with t-SNE visualization capability
        Args:
            feature_list {}:
            path {str}:
            experiment_name {}:
        """

        carry_on = (epoch <= 10) or ((epoch % 20) == 0)
        if not carry_on:
            return
        
        indices = np.random.choice(a=image.size()[0],
                                   size=int(num_samples),
                                   replace=False)

        image = image[indices].cpu().numpy().transpose(0, 2, 3, 1)
        feature_labels = attribute[indices].cpu().numpy()
        latents = self.store_z[indices].cpu().numpy()

        if feature_labels.dtype == int:
            feature_labels = np.array([str(x) for x in feature_labels])

        # t-SNE:
        if method == 'tsne':

            results = TSNE(
                n_components=2,
                verbose=1,
                metric='euclidean',
                perplexity=50,
                n_iter=1000,
                learning_rate=200).fit_transform(latents)
        else:
            results = umap.UMAP(n_neighbors=15,
                                min_dist=0.30,
                                metric='euclidean').fit_transform(latents)

        tx, ty = results[:, 0], results[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        path = os.path.expanduser(path)
        storage_path = f"{path}{run_name}/"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        if self.img_encoder.in_channels == 1:
            imgplot = 255 * np.ones((plot_size, plot_size), np.uint8)
        else:
            imgplot = 255 * \
                np.ones(
                    (plot_size,
                     plot_size,
                     self.img_encoder.in_channels),
                    np.uint8)

        # Fill the blank plot with the coordinates of the images according to
        # tSNE
        for img, label, x, y in tqdm(zip(image, feature_labels, tx, ty),
                                     desc='Plotting t-SNE/UMAP with images',
                                     total=len(image)):

            img = self.reshape_image(img, 20)
            tl_x, tl_y, br_x, br_y = self.get_coordinates(img, x, y, plot_size)

            # draw a rectangle with a color corresponding to the image class
            #image = draw_rectangle_by_class(img, label)
            if self.img_encoder.in_channels > 1:
                imgplot[tl_y:br_y, tl_x:br_x, :] = img
            else:
                imgplot[tl_y:br_y, tl_x:br_x] = img

        img_storage_path = f"{storage_path}/{method}img_{epoch}.png"
        cv2.imwrite(img_storage_path, imgplot)

        # plot scatterplot t-SNE results:
        plt.close()

        df = pd.DataFrame(
            {'x': results[:, 0], 'y': results[:, 1], 'category': feature_labels})
        num_unique_cats = len(df['category'].unique())
        palette = sns.color_palette("bright", num_unique_cats)
        fig = sns.relplot(
            x='x',
            y='y',
            data=df,
            hue='category',
            palette=palette,
            alpha=0.7)
        fig.savefig(f"{storage_path}/{method}_{epoch}.png")

    def _cluster_freq(self, path, run_name, epoch):
        try:
            self.store_probs * 1
        except:
            return

        carry_on = (epoch <= 10) or ((epoch % 20) == 0)
        if not carry_on:
            return
        
        storage_path = f"{path}{run_name}/"

        probs = copy.deepcopy(self.store_probs.cpu().numpy())
        probs2 = copy.deepcopy(self.store_probs.cpu().numpy())

        indices_max = np.argmax(probs, 1)
        one_hot_matrix = np.zeros((probs.shape[0], self.categorical_dim))
        for i in range(probs.shape[0]):
            one_hot_matrix[i,indices_max[i]] = 1
        
        cluster_percentages = one_hot_matrix.mean(0).round(4)
        cluster_names = ['cluster_{num}'.format(num=x) for x in range(self.categorical_dim+1)]
        mean_probability = probs2.mean(0)

        df = pd.DataFrame({'Distribution':cluster_percentages, 'Mean Probability':mean_probability}, index = cluster_names[1:])
        ax = df.plot.bar(rot=0)
        fig = ax.get_figure()
        fig.savefig(f"{storage_path}/cluster_distribution{epoch}.png")

    def get_coordinates(self, image, dimred_x, dimred_y, global_size):
        # changed from https://www.learnopencv.com/t-sne-for-feature-visualization/
        # Get height and width of image

        height, width, _ = image.shape

        # compute the image center coordinates for dimensionality reduction
        # plot
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
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        return resized



        

        


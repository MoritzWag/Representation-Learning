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

torch.set_default_dtype(torch.float64)


class Visualizer(nn.Module):

    def __init__(self, **kwargs):
        super(Visualizer, self).__init__(**kwargs)

    def _get_traversals(self, model_type, data=None):
        """
        Args:
            embedding: {torch} represents the
        """

        with torch.no_grad():
            image, attribute = next(iter(data))

            if torch.cuda.is_available():
                image, attribute = image.cuda(), attribute.cuda()

            if model_type == 'GaussmixVae':
            
                probabilities = torch.tensor(
                    [0.01, 0.05, 0.15, 0.25, 0.45, 0.5, 0.55, 0.75, 0.85, 0.95, 0.99]
                )

                num_latent_trav = probabilities.size()[0]
                
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
                for cat in range(self.categorical_dim):
                    for lat in range(self.latent_dim):
                        for prob in range(len(probabilities)):
                            mu = copy.deepcopy(self.mu_hat[cat])
                            mu[lat] = quantiles[cat, lat, prob]
                            traversal[total, :] = mu
                            total += 1

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
                for lat in range(self.latent_dim):
                    for prob in range(len(probabilities)):
                        mu = copy.deepcopy(self.mu_hat)
                        mu[lat] = quantiles[lat, prob]
                        traversal[total, :] = mu
                        total += 1

                if torch.cuda.is_available():
                    traversal = traversal.cuda()

            return traversal.float(), num_latent_trav

    def traversals(self,
                   epoch,
                   experiment_name,
                   path,
                   data=None):

        self.epoch = epoch

        carry_on = (epoch < 10) or ((epoch % 10) == 0)
        if not carry_on:
            return
        
        latent_traversals, n_per_latent = self._get_traversals(
            model_type = self.__class__.__name__,
            data = data
        )

        path = os.path.expanduser(path)
        storage_path = f"{path}{experiment_name}/"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        if self.__class__.__name__ == 'GaussmixVae':
            for categories in range(self.categorical_dim):

                decoded_traversals = self.img_decoder(
                    latent_traversals[categories]
                )

                vutils.save_image(
                    decoded_traversals.data,
                    f"{storage_path}traversal_{epoch}_cat{categories+1}.png",
                    normalize=True,
                    nrow=n_per_latent
                )
        else:
            decoded_traversals = self.img_decoder(
                latent_traversals
            )

            vutils.save_image(
                decoded_traversals.data,
                f"{storage_path}traversal_{epoch}.png",
                normalize=True,
                nrow=n_per_latent
            )

    def _sample_images(self,
                       val_gen,
                       epoch,
                       path,
                       experiment_name):

        carry_on = (epoch < 10) or ((epoch % 10) == 0)
        if not carry_on:
            return

        test_input, test_label = next(iter(val_gen))

        if torch.cuda.is_available():
            test_input, test_label = test_input.cuda(), test_label.cuda()

        if test_input.size()[0] > 32:
            indices = np.random.choice(test_input.size()[0], 32)
            test_input = test_input[indices]
            test_label = test_label[indices]

        path = os.path.expanduser(path)
        storage_path = f"{path}{experiment_name}/"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        reconstruction = self._generate(test_input)

        recon_real = torch.cat(
            (test_input.cuda(), reconstruction.type(
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

        del test_input, reconstruction

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

    def _cluster(self,
                 data,
                 path,
                 epoch,
                 experiment_name,
                 num_batches=10,
                 plot_size=1000):
        """Clustering algorithm with t-SNE visualization capability
        Args:
            feature_list {}:
            path {str}:
            experiment_name {}:
        """

        carry_on = (epoch < 10) or ((epoch % 10) == 0)
        if not carry_on:
            return

        indices = np.random.choice(a=len(data),
                                   size=int(num_batches),
                                   replace=False)

        features_extracted = []
        features_labels = []
        images_ = None

        for batch, (image, attribute) in enumerate(data):
            try:
                attribute = attribute[:, 0]
            except:
                pass

            if batch in indices:
                if torch.cuda.is_available():
                    image = image.cuda()
                h_enc = self.img_encoder(image.float())
                z = self._reparameterization(h_enc)
                z = z.cpu().detach().numpy()

                features_extracted.append(z)
                features_labels.append(attribute)
                if images_ is None:
                    images_ = image.cpu().numpy().transpose(0, 2, 3, 1)
                else:
                    image = image.cpu().numpy().transpose(0, 2, 3, 1)
                    images_ = np.append(images_, image, 0)
            else:
                pass

        features_extracted = np.vstack(features_extracted)
        features_labels = np.concatenate(features_labels)
        if features_labels.dtype == int:
            features_labels = np.array([str(x) for x in features_labels])

        # t-SNE:
        tsne_results = TSNE(
            n_components=2,
            verbose=1,
            metric='euclidean',
            perplexity=50,
            n_iter=1000,
            learning_rate=200).fit_transform(features_extracted)

        tx, ty = tsne_results[:, 0], tsne_results[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        path = os.path.expanduser(path)
        storage_path = f"{path}{experiment_name}/"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        if self.img_encoder.in_channels == 1:
            tsne_imgplot = 255 * np.ones((plot_size, plot_size), np.uint8)
        else:
            tsne_imgplot = 255 * \
                np.ones(
                    (plot_size,
                     plot_size,
                     self.img_encoder.in_channels),
                    np.uint8)

        # Fill the blank plot with the coordinates of the images according to
        # tSNE
        for img, label, x, y in tqdm(zip(images_, features_labels, tx, ty),
                                     desc='Plotting t-SNE with images',
                                     total=len(images_)):

            img = self.reshape_image(img, 25)
            tl_x, tl_y, br_x, br_y = self.get_coordinates(img, x, y, plot_size)

            # draw a rectangle with a color corresponding to the image class
            #image = draw_rectangle_by_class(img, label)
            if self.img_encoder.in_channels > 1:
                tsne_imgplot[tl_y:br_y, tl_x:br_x, :] = img
            else:
                tsne_imgplot[tl_y:br_y, tl_x:br_x] = img

        img_storage_path = f"{storage_path}/clusterimg_{epoch}.png"
        cv2.imwrite(img_storage_path, tsne_imgplot)

        # plot scatterplot t-SNE results:
        plt.close()

        df = pd.DataFrame(
            {'x': tsne_results[:, 0], 'y': tsne_results[:, 1], 'category': features_labels})
        num_unique_cats = len(df['category'].unique())
        palette = sns.color_palette("bright", num_unique_cats)
        fig = sns.relplot(
            x='x',
            y='y',
            data=df,
            hue='category',
            palette=palette,
            alpha=0.7)
        fig.savefig(f"{storage_path}/cluster_{epoch}.png")

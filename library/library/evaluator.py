import pandas as pd 
import numpy as np 
import sklearn as sk 
from sklearn.neighbors import KNeighborsRegressor
import scipy 

from library.eval_helpers import make_discretizer, discrete_mutual_info

class Evaluator(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)

    def _knn_regressor(self, data):

        zs = []
        for batch, (image, attribute) in enumerate(data):
            h_enc = self.img_encoder(image.float())
            z = self._reparameterization(h_enc)
            z = z.cpu().detach().cpu()
            zs.append(z)
            
            # work with the labels here as well!

        
        ## train

        ## predict 

        ## evaluate
    
    def unsupervised_metrics(self, data):
        """
        """
        scores = {}
        
        zs = []
        for batch, (image, attribute) in enumerate(data):
            h_enc = self.img_encoder(image.float())
            z = self._reparameterization(h_enc)
            z = z.cpu().detach().cpu()
            zs.append(z)
        
        zs = np.stack(zs)
        zs = np.squeeze(zs)

        num_latents = zs.shape[1]
        cov_zs = np.cov(zs)

        scores['gaussian_total_correlation'] = gaussian_total_correlation(cov_zs)
        scores['gaussian_wasserstein_correlation'] = guassian_wasserstein_correlation(cov_zs)
        scores['gaussian_wasserstein_correlation_norm'] = (
                scores['gaussian_wasserstein_correlation'] / np.sum(np.diag(cov_zs)))


        zs_discrete = make_discretizer(zs)
        mutual_info_matrix = discrete_mutual_info(mus_discrete, mus_discrete)
        np.fill_diagonal(mutual_info_matrix, 0)
        mutual_info_score = np.sum(mutual_info_matrix) / (num_latents**2 - num_latents)
        scores['mutual_info_score'] = mutual_info_score

        return scores 

    def gaussian_total_correlation(self, cov):
        """
        """
        return 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])
    
    def gaussian_wasserstein_correlation(self, cov):
        """
        """
        sqrtm = scipy.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
        return 2 * np.trace(cov) - 2 * np.trace(sqrtm)
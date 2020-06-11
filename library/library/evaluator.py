import pandas as pd 
import numpy as np 
import sklearn as sk 
from sklearn.neighbors import KNeighborsRegressor
import scipy 

from library.eval_helpers import make_discretizer, discrete_mutual_info, knn_regressor

class Evaluator(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)
        self.scores = {}

    def _downstream_task(self, train_data, test_data, function='knn_regressor'):
        


        zs_train = []
        for batch, (image, attribute) in enumerate(train_data):
            h_enc = self.img_encoder(image.float())
            z = self._reparameterization(h_enc)
            z = z.cpu().detach().cpu()
            zs_train.append(z)
        
        zs_test = []
        for batch, (image, attribute) in enumerate(test_data):
            h_enc = self.img_encoder(image.float())
            z = self._reparameterization(h_enc)
            z = z.cpu().detach().cpu()
            zs_test.append(z)

        zs_train = np.stack(zs_train)
        zs_train = np.squeeze(zs_train)

        zs_test = np.stack(zs_test)
        zs_test = np.squeeze(zs_test)

        # work with the labels here as well!
        if function == 'knn_regressor':
            mse, mae = knn_regressor()
            self.scores['downstream_task_mse'] = mse
            self.scores['downstream_task_mae'] = mae
        elif function == 'knn_classifier':
            acc, auc = knn_classifier()
            self.scores['downstream_task_acc'] = acc
            self.scores['downstream_task_auc'] = auc
    
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

        self.scores['gaussian_total_correlation'] = gaussian_total_correlation(cov_zs)
        self.scores['gaussian_wasserstein_correlation'] = gaussian_wasserstein_correlation(cov_zs)
        self.scores['gaussian_wasserstein_correlation_norm'] = (
                self.scores['gaussian_wasserstein_correlation'] / np.sum(np.diag(cov_zs)))


        #zs_discrete = make_discretizer(zs)
        zs_discrete = histogram_discretize(zs)   
        mutual_info_matrix = discrete_mutual_info(mus_discrete, mus_discrete)
        np.fill_diagonal(mutual_info_matrix, 0)
        mutual_info_score = np.sum(mutual_info_matrix) / (num_latents**2 - num_latents)
        self.scores['mutual_info_score'] = mutual_info_score
 

    def gaussian_total_correlation(self, cov):
        """
        """
        return 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])
    
    def gaussian_wasserstein_correlation(self, cov):
        """
        """
        sqrtm = scipy.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
        return 2 * np.trace(cov) - 2 * np.trace(sqrtm)
    

    def log_metrics(self):
        """
        """
        self.scores
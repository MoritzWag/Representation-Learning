import pandas as pd 
import numpy as np 
import sklearn as sk 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
import scipy 
import pdb
import torch
import csv
import os
from torch import nn, optim, Tensor
from torch.distributions import MultivariateNormal
from torch.distributions import kl_divergence

from library.eval_helpers import histogram_discretize, discrete_mutual_info, knn_regressor, knn_classifier, random_forest
from library.utils import accumulate_batches


class Evaluator(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)
        self.scores = {}

    def _downstream_task(self, train_data, test_data, model, storage_path, downstream_task_names=None):

        features_train, target_train = train_data[0].cpu().numpy(), train_data[1].cpu().numpy()
        features_test, target_test = test_data[0].cpu().numpy(), test_data[1].cpu().numpy()

        if downstream_task_names is None:
            downstream_task_names = 'downstream task ' * target_train.shape[1]

        if model == 'random_forest':
            if target_train.shape[1] > 1:
                
                feature_imp = []

                for i in range(target_train.shape[1]):
                    acc, auc, aupr, avg_pr, importances = random_forest(
                        features_train,
                        target_train[:,i],
                        features_test,
                        target_test[:,i],
                        downstream_task_names[i],
                        storage_path
                    )

                    self.scores['rf_acc'+downstream_task_names[i]] = acc.round(5)
                    self.scores['rf_auc'+downstream_task_names[i]] = auc
                    self.scores['rf_aupr'+downstream_task_names[i]] = aupr.round(5)
                    self.scores['rf_avg_pr'+downstream_task_names[i]] = avg_pr.round(5)

            else:
                acc, auc, aupr, avg_pr = random_forest(
                        features_train,
                        target_train[:,i],
                        features_test,
                        target_test[:,i],
                        'downstream task',
                        storage_path
                    )
        
                self.scores['dst_rf_acc'] = acc.round(5)
                self.scores['dst_rf_auc'] = auc
                self.scores['dst_rf_aupr'] = aupr.round(5)
                self.scores['dst_rf_avg_pr'] = avg_pr.round(5)

        elif model == 'knn_classifier':
            if target_train.shape[1] > 1:
            
                for i in range(target_train.shape[1]):
                    acc = knn_classifier(features_train, target_train[:,i], features_test, target_test[:,i])

                    self.scores['knn_acc'+downstream_task_names[i]] = acc.round(5)
            else:
                acc = knn_classifier(features_train, target_train, features_test, target_test)
                self.scores['dst_knn_acc'] = acc.round(5)
    
    def unsupervised_metrics(self, data):
        """
        """        
        
        zs = data.cpu().numpy()
        num_latents = zs.shape[1]
        zs_transposed = np.transpose(zs)
        cov_zs = np.cov(zs_transposed)

        self.scores['gaussian_total_correlation'] = self.gaussian_total_correlation(cov_zs)
        self.scores['gaussian_wasserstein_correlation'] = self.gaussian_wasserstein_correlation(cov_zs)
        self.scores['gaussian_wasserstein_correlation_norm'] = (
                self.scores['gaussian_wasserstein_correlation'] / np.sum(np.diag(cov_zs)))


        #features_discrete = make_discretizer(zs)
        features_discrete = histogram_discretize(zs, num_bins=400)   
        mutual_info_matrix = discrete_mutual_info(features_discrete, features_discrete)
        np.fill_diagonal(mutual_info_matrix, 0)
        mutual_info_score = np.sum(mutual_info_matrix) / (num_latents**2 - num_latents)
        self.scores['mutual_info_score'] = mutual_info_score

    def mutual_information(self, latent_loss):
        
        z = self.store_z.transpose(1,0).cpu().numpy()
        z_mean = z.mean(-1)
        z_cov = np.cov(z)
        p_z = MultivariateNormal(loc=torch.tensor(z_mean), covariance_matrix=torch.tensor(z_cov))
        q_z = MultivariateNormal(loc=torch.zeros(z_mean.shape[0]), covariance_matrix=torch.eye(z_mean.shape[0]))

        kldiv_priors = kl_divergence(p_z, q_z)
        
        return latent_loss - kldiv_priors
 
    def gaussian_total_correlation(self, cov):
        """
        """
        return 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])
    
    def gaussian_wasserstein_correlation(self, cov):
        """
        """
        sqrtm = scipy.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
        return 2 * np.trace(cov) - 2 * np.trace(sqrtm)
    

    def log_metrics(self, storage_path):
        """
        """

        if storage_path is not None:
            if not os.path.exists(storage_path):
                os.makedirs(f"./{storage_path}")
            
            df = pd.DataFrame(self.scores, index=[0])
            df.to_csv(f"{storage_path}eval_metrics.csv")

import pandas as pd 
import numpy as np 
import sklearn as sk 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
import scipy 
import pdb
import torch
import csv
import os
from torch import nn, optim, Tensor

from library.eval_helpers import histogram_discretize, discrete_mutual_info, knn_regressor, knn_classifier
from library.utils import accumulate_batches


class Evaluator(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)
        self.scores = {}

    def _downstream_task(self, train_data, test_data, function):
        
        features_train, target_train = train_data[0].cpu().numpy(), train_data[1].cpu().numpy()
        features_test, target_test = test_data[0].cpu().numpy(), test_data[1].cpu().numpy()
            
        if target_train.shape[1] > 1:
            
            for i in range(target_train.shape[1]):
                acc = knn_classifier(features_train, target_train[:,i], features_test, target_test[:,i])
                self.scores['downstream_task_acc'+str(i+1)] = acc.round(5)
        else:
            acc = knn_classifier(features_train, target_train, features_test, target_test)
            self.scores['downstream_task_acc'] = acc.round(5)
    
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

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

class Evaluator(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)
        self.scores = {}

    def _downstream_task(self, train_data, test_data, function, column_index):
        if len(column_index) == 1:
            features_train = []
            target_train = []
            for batch, (image, attribute) in enumerate(train_data):
                attribute = attribute[:, column_index]
                if torch.cuda.is_available():
                    image = image.cuda()
                h_enc = self.img_encoder(image.float())
                z = self._reparameterization(h_enc)
                z = z.cpu().detach().numpy()
                features_train.append(z)
                target_train.append(attribute)
                        
            features_test = []
            target_test = []
            for batch, (image, attribute) in enumerate(test_data):
                attribute = attribute[:, column_index]
                if torch.cuda.is_available():
                    image = image.cuda()
                h_enc = self.img_encoder(image.float())
                z = self._reparameterization(h_enc)
                z = z.cpu().detach().numpy()
                features_test.append(z)
                target_test.append(attribute)

            features_train = np.vstack(features_train)
            target_train = np.concatenate(target_train)

            features_test = np.vstack(features_test)
            target_test = np.concatenate(target_test)

        else:
            features_train = []
            target_train = []
            for batch, (image, attribute) in enumerate(train_data):
                attribute, z = attribute[:, column_index].transpose(0,1)
                features_train.append(z)
                target_train.append(attribute)
                        
            features_test = []
            target_test = []
            for batch, (image, attribute) in enumerate(test_data):
                attribute, z = attribute[:, column_index].transpose(0,1)
                features_test.append(z)
                target_test.append(attribute)

            features_train_int = np.concatenate(features_train)
            features_test_int = np.concatenate(features_test)
            features_total = np.append(features_train_int, features_test_int)

            features_total_chr = np.array([str(x) for x in features_total])

            enc = OneHotEncoder()
            enc.fit(features_total_chr.reshape(-1, 1))
            features_total_oh = enc.transform(features_total_chr.reshape(-1, 1)).toarray()

            features_train = features_total_oh[range(len(features_train_int)), :]
            features_test = features_total_oh[-len(features_test_int):, :]

            target_train = np.concatenate(target_train)
            target_test = np.concatenate(target_test)

        # work with the labels here as well!
        if function == 'knn_regressor':
            if len(column_index) == 1:
                mse, mae = knn_regressor(features_train, target_train, features_test, target_test)
                self.scores['downstream_task_mse'] = mse.round(3)
                self.scores['downstream_task_mae'] = mae.round(3)
            else:
                mse, mae = knn_regressor(features_train, target_train, features_test, target_test)
                self.scores['baseline_mse'] = mse.round(3)
                self.scores['baseline_mae'] = mae.round(3)
        elif function == 'knn_classifier':
            acc = knn_classifier(features_train, target_train, features_test, target_test)
            self.scores['downstream_task_acc'] = acc.round(5)
            #self.scores['downstream_task_auc'] = auc
        else: 
            pass
    
    def unsupervised_metrics(self, data):
        """
        """        
        zs = []
        for batch, (image, attribute) in enumerate(data):
            if torch.cuda.is_available():
                    image = image.cuda()
            h_enc = self.img_encoder(image.float())
            z = self._reparameterization(h_enc)
            z = z.cpu().detach().cpu()
            zs.append(z)
        
        zs = np.vstack(zs)
        zs = np.squeeze(zs)

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

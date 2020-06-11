import numpy as np 
import pandas as pd 
import sklearn as sk 
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KNeighborsRegressor


def make_discretizer():
    pass

def discrete_mutual_info(zs, ys):
    """Computes discrete mutual information
    """
    num_codes = zs.shape[1]
    num_factors = ys.shape[1]

    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = mutual_info_score(ys[:, j], zs[:, i])
    
    return m 

def discrete_entropy(zs):
    """Computes discretized entropy
    """
    num_factors = zs.shape[1]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = mutual_info_score(zs[:, j], zs[:, j])
    
    return h 

def histogram_discretize(target, num_bins):
    """Discretiation based on histograms
    """
    discretized = np.zeros_like(target)
    for i in range(target.shape[1]):
        discretized[:, i] = np.digitize(target[:, i],
                                np.histogram(
                                    target[:, i], num_bins
                                )[1][:, -1])
    
    return discretized


def knn_regressor(train_data, test_data):
    
    knn = KNeighborsRegressor()
    knn.fit()

    return mse, mae 
def knn_classifier():
    pass



def save_metrics(scores, save_path, epoch=None):
    """Rearranges scores dictionary to pandas.DataFrame
    """
    if epoch is None:
        pass 
    else:
        pass
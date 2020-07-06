import numpy as np 
import pandas as pd 
import sklearn as sk 
import pdb
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neibors import KNeighborsClassifier


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
                                )[1][:-1])
    
    return discretized


def knn_regressor(train_X, train_y, test_X, test_y):
    
    knn = KNeighborsRegressor(n_neighbors = 5)
    knn.fit(train_X, train_y)
    pred_y = knn.predict(test_X)
    mse = ((pred_y - test_y)**2).mean()
    mae = abs((pred_y - test_y)).mean()

    return mse, mae 

def knn_classifier(train_X, train_y, test_X, test_y):
    
    knn = KNeighborsClassifier(n_neighbors = 10)
    knn.fit(train_X, train_y)
    acc = knn.score(test_X, test_y)
    
    return acc

def save_metrics(scores, save_path, epoch=None):
    """Rearranges scores dictionary to pandas.DataFrame
    """
    if epoch is None:
        pass 
    else:
        pass


def random_forest(features, labels):
    """
    """
    pdb.set_trace()
    model = RandomForestClassifier(class_weight="balanced")
    model.fit(features, labels)
    importance = model.feature_importances

    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

    yhat = model.predict()

    

    return yhat
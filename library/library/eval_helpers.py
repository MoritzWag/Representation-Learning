import numpy as np 
import pandas as pd 
import sklearn as sk 
import matplotlib.pyplot as plt
import pdb
import os
from sklearn.metrics import mutual_info_score, balanced_accuracy_score, roc_auc_score, average_precision_score, precision_recall_curve, auc
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelBinarizer
from scipy import sparse
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
    
    knn = KNeighborsClassifier(n_neighbors = 5).fit(train_X, train_y)
    acc = knn.score(test_X, test_y)
    
    return acc

def save_metrics(scores, save_path, epoch=None):
    """Rearranges scores dictionary to pandas.DataFrame
    """
    if epoch is None:
        pass 
    else:
        pass


def random_forest(train_X, train_y, test_X, test_y, dst_name, path):
    """
    """
    # Create one-hot-encoding
    print("one hot encoding")
    y = np.concatenate((train_y, test_y))
    ybin = LabelBinarizer().fit(y).transform(y)
    train_yb = ybin[:train_y.shape[0], :]
    test_yb = ybin[train_y.shape[0]:, :]

    sparse_cat_test = test_yb.sum(axis=0) != 0 
    sparse_cat_train = train_yb.sum(axis=0) != 0
    drop_index = sparse_cat_train == sparse_cat_test
    train_yb = train_yb[:, drop_index]
    test_yb = test_yb[:, drop_index]

    cats = np.union1d(np.unique(test_y), np.unique(train_y))
    drop_cat = cats[np.where(drop_index==False)[0]]
    test_y_new = pd.Series(test_y)
    train_y_new = pd.Series(train_y)
    test_y = test_y_new[~test_y_new.isin(drop_cat)]
    test_X = test_X[~test_y_new.isin(drop_cat),:]
    train_y = train_y_new[~train_y_new.isin(drop_cat)]
    train_X = train_X[~train_y_new.isin(drop_cat),:]

    y = np.concatenate((train_y, test_y))
    ybin = LabelBinarizer().fit(y).transform(y)
    train_yb = ybin[:train_y.shape[0], :]
    test_yb = ybin[train_y.shape[0]:, :]

    print("initiate model")
    # Instatiate model
    rf = RandomForestClassifier(
        class_weight="balanced",
        min_samples_leaf=10,
        n_estimators=100).fit(train_X, train_y) # min samples leaf is based on the mean of the categories of product type descr that have less than 50 counts
    
    # Obtain Metrics
    print("obtain metrics")
    probs_hat = rf.predict_proba(test_X)
    y_hat = rf.predict(test_X)

    #pdb.set_trace()
    try:
        weighted_auc = roc_auc_score(y_true=test_yb, y_score=probs_hat, multi_class="ovr", average='weighted')
    except:
        weighted_auc = 0.0
    balanced_acc = balanced_accuracy_score(test_y, y_hat)

    try:
        avg_precision = average_precision_score(test_yb, probs_hat, average='weighted')
    except: 
        avg_precision = np.asarray(0.0)

    prevalence = ybin.sum(axis=0) / ybin.sum()
    aupr = []

    print("blabblub")
    for i in range(len(prevalence)):
        precision, recall, thresholds = precision_recall_curve(test_yb[:,i], probs_hat[:,i])
        auc_ = auc(recall, precision)
        aupr.append(auc_)
    
    aupr = np.array(aupr)
    aupr_wmean = (aupr * prevalence).sum() 
    
    print("get feature importance")
    #Obtain permutation importance
    importance = permutation_importance(rf, test_X, test_y, n_repeats=15, random_state=0)
    sorted_idx = importance.importances_mean.argsort()

    fig, ax = plt.subplots()
    latent_names = ['latent_{num}'.format(num=x) for x in (sorted_idx+1)]
    #latent_names = latent_names[1:]
    ax.boxplot(importance.importances[sorted_idx].T,
           vert=False, labels=latent_names)
    ax.set_title("Permutation Importances "+str(dst_name))
    fig.tight_layout()

    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}FI_{dst_name}.png") 

    plt.close()

    return balanced_acc, weighted_auc, aupr_wmean, avg_precision
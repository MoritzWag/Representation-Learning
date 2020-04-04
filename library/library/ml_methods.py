import numpy as np 
import pandas as pd 
import sklearn as sk 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



class TSNE(object):
    """
    """
    def __init__(self):
        pass

    def transform(self, x, n_components):
        transformed_x = TSNE(n_components=n_components).fit_transform(x)
        return transformed_x

    def visualize(self, x, label, path, num_obs):
        x = x[:num_obs]
        y = y[:num_obs]
        colors = 
        for i, c, label in zip(ta):
            plt.scatter(x[y == i, 0], x[y == 1, y], c=c, label=label))
        plt.legend()
        plt.show()


class PCA(object):
    """
    """
    def __init__(self):
        pass

    def transform(self, x):
        pass

    def visualize(self, x):
        pass 

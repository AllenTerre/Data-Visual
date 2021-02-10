#!/usr/bin/python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA, PCA, FastICA, NMF
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import TruncatedSVD



def PCAProjection(myArray, dim):
    X_transformed = TruncatedSVD(n_components=dim).fit_transform(myArray)
    return X_transformed

def LLEProjection(myArray, dim):
    embedding = LocallyLinearEmbedding(n_components=dim)
    X_transformed = embedding.fit_transform(myArray)
    return X_transformed 

def ISOMAPProjection(myArray, dim):
    embedding = Isomap(n_components=dim)
    X_transformed = embedding.fit_transform(myArray)
    return X_transformed

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target           # Use for distinguish different species. 







def scatter_single_2D(array, method='pca', color=None):
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(array[:, 0], array[:, 1], c=color, cmap=plt.cm.Paired,
            edgecolor='k')
    
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_xlabel("1st eigenvector")
    ax.set_ylabel("2nd eigenvector")
    ax.set_title("Keep 2 features using: "+ method)
    plt.show()
    
    
    
def scatter_plot_2D_pair(array):
    fig = plt.figure(12, figsize = (15, 10))
    ax1 = fig.add_subplot(221)
    ax1.scatter(array[:, 0], array[:, 1], c=y, cmap=plt.cm.Paired,
            edgecolor='k')
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.set_title("1st & 2nd eigenvector")
    
    ax2 = fig.add_subplot(222)
    ax2.scatter(array[:, 0], array[:, 2], c=y, cmap=plt.cm.Paired,
            edgecolor='k')
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    ax2.set_title("1st & 3rd eigenvector")
    
    ax3 = fig.add_subplot(212)
    ax3.scatter(array[:, 1], array[:, 2], c=y, cmap=plt.cm.Paired,
            edgecolor='k')
    
    ax3.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])
    ax3.set_title("2st & 3rd eigenvector")
    fig.show()
    
    
    
    
def scatter_plot_3D(array, method = 'PCA', color=None):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(array[:, 0], array[:, 1], array[:, 2], c=color,
           cmap=plt.cm.Paired,marker = 'o', edgecolor='k', s=40)
    ax.set_title("Keep 3 features in the dataset using " + method)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()

    


if __name__ == "__main__":
    pca_3 = PCAProjection(iris.data,dim=3)
    pca_2 = PCAProjection(iris.data,dim=2)
    scatter_plot_3D(pca_3, 'PCA',y)
    scatter_single_2D(pca_2, 'PCA',y)
    
    lle_3 = LLEProjection(iris.data,dim=3)
    lle_2 = LLEProjection(iris.data,dim=2)
    scatter_plot_3D(lle_3, 'LLE',y)
    scatter_single_2D(lle_2, 'LLE',y)
    
    iso_3 = ISOMAPProjection(iris.data,dim=3)
    iso_2 = ISOMAPProjection(iris.data,dim=2)
    scatter_plot_3D(iso_3, 'ISO',y)
    scatter_single_2D(iso_2, 'ISO',y)
    
    
    
    
    
    
    
    
    
    
    
    
    
    




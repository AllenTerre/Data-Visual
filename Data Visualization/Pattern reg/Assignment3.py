#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:15:24 2020

"""

import numpy as np
#import visualization
import matplotlib.pyplot as plt
#from scipy.stats import norm, multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


# spilt data set 

file = open('Assignment3_dataset.txt','r')
ls = []
for line in file:
    line = line.strip('\n')  
    ls.append(line.split( ))   


w1 = np.array(ls[1:11]).astype(np.float32)  # orginal data

#w2 = np.array(ls[12:22]).astype(np.float32)
#w3 = np.array(ls[23:]).astype(np.float32)


'''

GM = GaussianMixture(n_components = 3, 
                      covariance_type = 'full', 
                      max_iter = 600, random_state = 3)


GM.means_ = np.zeros(3)
GM.covariances_ = np.identity(3)
GM.fit(w1)
cluster = GM.predict(w1)
cluster_p = np.round(GM.predict_proba(w1),3)
accuracy = silhouette_score(w1, cluster)

weights = GM.weights_
means = GM.means_
covariances = GM.covariances_

'''


def EM_Process(data, n, covt):
    
    '''
    data: array shape data. 
    n: the number of components
    covt: covariance_type 
    {‘full’, ‘tied’, ‘diag’, ‘spherical’}
    chose one of them. 
    
    '''
    
    GM = GaussianMixture(n_components = n, 
                      covariance_type = covt, 
                      max_iter = 600, random_state = 3)
    GM.means_ = np.zeros(3)
    GM.covariances_ = np.identity(3)
    GM.fit(data)
    clusters = GM.predict(data)
    
    return clusters





def draw_3d_plot(array, cluster):
    x = array[:,0]
    y = array[:,1]
    z = array[:,2]
    fig = plt.figure(figsize=(6,4))
    ax = Axes3D(fig)
    sc = ax.scatter(x, y, z, s=40, c=cluster, alpha=1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()



def proba(true_data, guess_data):
    base = len(true_data)
    count = 0
    for i in range(0, base):
        if true_data[i] == guess_data[i]:
            count += 1 
            
    proba = count / base 
    
    return proba 



if __name__ == "__main__":
    x1 = w1[: ,0]
    x2 = w1[:, 1]
    x3_miss = np.divide(x1+x2, 2)     # assume x3 is missing 

    data_miss = np.vstack((x1,x2,x3_miss)).T
    data = w1 
    
    cluster_miss = EM_Process(data_miss, 3, 'diag')
    draw_3d_plot(data_miss, cluster_miss)
    
    cluster_true = EM_Process(data, 3, 'diag')
    draw_3d_plot(data, cluster_true)
    
    proba = proba(cluster_true, cluster_miss)
    print("The samilarity of two methods is: ", proba)









#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:35:37 2020

@author: zhoushining
"""

import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.neighbors import KernelDensity




data = np.array([0,1,1,1,2,2,2,2,3,4,4,4,5])

bins = [0,1,2,3,4,5]

kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(data[:, None])
logprob = kde.score_samples(data[:, None])
edata = np.exp(logprob)

'''

plt.fill_between(data, np.exp(logprob), alpha=0.5)
plt.plot(data, np.full_like(data, -0.01), '|k', markeredgewidth=1)
plt.show()

'''

plt.hist(data,bins=bins)
plt.title("histogram")
plt.show()



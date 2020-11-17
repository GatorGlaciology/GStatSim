#!/usr/bin/env python
# coding: utf-8

# In[12]:


### geostatistical tools
# Mickey MacKie


# In[1]:


import numpy as np
import numpy.linalg as linalg
import pandas as pd
import sklearn as sklearn
from sklearn.neighbors import KDTree
import math
from scipy.spatial import distance_matrix
from tqdm import tqdm


# In[2]:


# covariance function definition
def covar(t, d, r):
    h = d / r
    if t == 1:  # Spherical
        c = 1 - h * (1.5 - 0.5 * np.square(h))
        c[h > 1] = 0
    elif t == 2:  # Exponential
        c = np.exp(-3 * h)
    elif t == 3:  # Gaussian
        c = np.exp(-3 * np.square(h))
    return c



# get variogram along the major or minor axis
def axis_var(lagh, nug, nstruct, cc, vtype, a):
    lagh = lagh
    nstruct = nstruct # number of variogram structures
    vtype = vtype # variogram types (Gaussian, etc.)
    a = a # range for axis in question
    cc = cc # contribution of each structure
    
    n = len(lagh)
    gamma_model = np.zeros(shape = (n))
    
    # for each lag distance
    for j in range(0,n):
        c = nug
        c = 0
        h = np.matrix(lagh[j])
        
        # for each structure in the variogram
        for i in range(nstruct):
            Q = h.copy()
            d = Q / a[i]
            c = c + covar(vtype[i], d, 1) * cc[i] # covariance
        
        gamma_model[j] = 1+ nug - c # variance
    return gamma_model



# make array of x,y coordinates based on corners and resolution
def pred_grid(xmin, xmax, ymin, ymax, pix):
    cols = (xmax - xmin)/pix; rows = (ymax - ymin)/pix  # number of rows and columns
    x = np.arange(xmin,xmax,pix); y = np.arange(ymin,ymax,pix) # make arrays

    xx, yy = np.meshgrid(x,y) # make grid
    yy = np.flip(yy) # flip upside down

    # shape into array
    x = np.reshape(xx, (int(rows)*int(cols), 1))
    y = np.reshape(yy, (int(rows)*int(cols), 1))

    Pred_grid_xy = np.concatenate((x,y), axis = 1) # combine coordinates
    return Pred_grid_xy



# rotation matrix (Azimuth = major axis direction)
def Rot_Mat(Azimuth, a_max, a_min):
    theta = (Azimuth / 180.0) * np.pi
    Rot_Mat = np.dot(
        np.array([[1 / a_max, 0], [0, 1 / a_min]]),
        np.array(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        ),
    )
    return Rot_Mat



# covariance model
def cov(h1, h2, k, vario):
    # unpack variogram parameters
    Azimuth = vario[0]
    nug = vario[1]
    nstruct = vario[2]
    vtype = vario[3]
    cc = vario[4]
    a_max = vario[5]
    a_min = vario[6]
    
    c = -nug # nugget effect is made negative because we're calculating covariance instead of variance
    for i in range(nstruct):
        Q1 = h1.copy()
        Q2 = h2.copy()
        
        # covariances between measurements
        if k == 0:
            d = distance_matrix(
                np.matmul(Q1, Rot_Mat(Azimuth, a_max[i], a_min[i])),
                np.matmul(Q2, Rot_Mat(Azimuth, a_max[i], a_min[i])),
            )
            
        # covariances between measurements and unknown
        elif k == 1:
            d = np.sqrt(
                np.square(
                    (np.matmul(Q1, Rot_Mat(Azimuth, a_max[i], a_min[i])))
                    - np.tile(
                        (
                            np.matmul(
                                Q2, Rot_Mat(Azimuth, a_max[i], a_min[i])
                            )
                        ),
                        (k, 1),
                    )
                ).sum(axis=1)
            )
            d = np.asarray(d).reshape(len(d))
        c = c + covar(vtype[i], d, 1) * cc[i]
    return c



# simple kriging
def krige(Pred_grid, df, xx, yy, data, k, vario):
    
    Mean_1 = np.average(df[data]) # mean of data
    Var_1 = np.var(df[data]); # variance of data 
    
    # make KDTree to search data for nearest neighbors
    tree_data = KDTree(df[[xx,yy]].values) 
    
    # preallocate space for mean and variance
    est_SK = np.zeros(shape=len(Pred_grid))
    var_SK = np.zeros(shape=len(Pred_grid))
    
    X_Y = np.zeros((1, k, 2))
    closematrix_Primary = np.zeros((1, k))
    neardistmatrix = np.zeros((1, k))
    
    for z in tqdm(range(0, len(Pred_grid))):
        # find nearest data points
        nearest_dist, nearest_ind = tree_data.query(Pred_grid[z : z + 1, :], k=k)
        a = nearest_ind.ravel()
        group = df.iloc[a, :]
        closematrix_Primary[:] = group[data]
        neardistmatrix[:] = nearest_dist
        X_Y[:, :] = group[[xx, yy]]
        
        # left hand side (covariance between data)
        Kriging_Matrix = np.zeros(shape=((k, k)))
        Kriging_Matrix = cov(X_Y[0], X_Y[0], 0, vario)
        
        # Set up Right Hand Side (covariance between data and unknown)
        r = np.zeros(shape=(k))
        k_weights = r
        r = cov(X_Y[0], np.tile(Pred_grid[z], (k, 1)), 1, vario)
        Kriging_Matrix.reshape(((k)), ((k)))
        
        # Calculate Kriging Weights
        k_weights = np.dot(np.linalg.pinv(Kriging_Matrix), r)

        # get estimates
        est_SK[z] = k*Mean_1 + np.sum(k_weights*(closematrix_Primary[:] - Mean_1))
        var_SK[z] = Var_1 - np.sum(k_weights*r)
        
    return est_SK, var_SK


# In[ ]:





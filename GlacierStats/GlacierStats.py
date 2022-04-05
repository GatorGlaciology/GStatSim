#!/usr/bin/env python
# coding: utf-8

# In[5]:


### geostatistical tools


# In[3]:

import nvtx
import cupy as cp
import cudf
from cuml.metrics import pairwise_distances
import numpy as np
import numpy.linalg as linalg
import pandas as pd
import sklearn as sklearn
from sklearn.neighbors import KDTree
import math
from scipy.spatial import distance_matrix
from tqdm import tqdm
import random


# In[4]:

def helloworld():
    print("hello world")


@nvtx.annotate("covar()", color="purple")
# covariance function definition
def covar(t, d, r):
    h = d / r
    if t == 1:  # Spherical
        c = 1 - h * (1.5 - 0.5 * cp.square(h))
        c[h > 1] = 0
    elif t == 2:  # Exponential
        c = cp.exp(-3 * h)
    elif t == 3:  # Gaussian
        c = cp.exp(-3 * cp.square(h))
    return c


@nvtx.annotate("sortQuadrantPoints()", color="red")
def sortQuadrantPoints(quad_array, quad_count, rad):
    quad_array['Dist'] = cp.linalg.norm(quad_array[["X","Y"]].values, axis = 1)
    quad_array = quad_array[quad_array.Dist < rad] # delete points outside of radius
    quad_array = quad_array.sort_values('Dist',ascending = True) # sort array by distances
    # select the number of points in each quadrant up to our quadrant count
    smallest = quad_array.iloc[:quad_count]
    return smallest
 
@nvtx.annotate("nearestNeighborSearch()", color="blue")
def nearestNeighborSearch(rad, count, loc, data):
    locx = loc[0]
    locy = loc[1]
    
    # wipe coords for re-usability 
    #coords = np.empty_like(data)
    coords = data.copy()
    #coords[:] = data
    # standardize our quadrants (create the origin at our location point)
    coords.X -= locx
    coords.Y -= locy
    
    # Number of points to look for in each quadrant, if not fully divisible by 4, round down
    quad_count = count//4
    
    # sort coords of dataset into 4 quadrants relative to input location
    final_quad = []
    final_quad.append(coords[(coords.X >= 0) & (coords.Y >= 0)])
    final_quad.append(coords[(coords.X < 0) & (coords.Y < 0)])
    final_quad.append(coords[(coords.X >= 0) & (coords.Y < 0)])
    final_quad.append(coords[(coords.X < 0) & (coords.Y >= 0)])
    
    # Gather distance values for each coord from point and delete points outside radius
    fcoord = []
    for quad in final_quad:
        fcoord.append(sortQuadrantPoints(quad, quad_count, rad))
    
    # add all quadrants back together for final dataset
    near = cudf.concat(fcoord)
    # unstandardize data back to original form
    near.X += locx
    near.Y += locy
    return near



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
@nvtx.annotate("pred_grid()", color="green")
def pred_grid(xmin, xmax, ymin, ymax, pix):
    cols = np.rint((xmax - xmin)/pix); rows = np.rint((ymax - ymin)/pix)  # number of rows and columns
    x = np.arange(xmin,xmax,pix); y = np.arange(ymin,ymax,pix) # make arrays

    xx, yy = np.meshgrid(x,y) # make grid
    yy = np.flip(yy) # flip upside down

    # shape into array
    x = np.reshape(xx, (int(rows)*int(cols), 1))
    y = np.reshape(yy, (int(rows)*int(cols), 1))

    Pred_grid_xy = np.concatenate((x,y), axis = 1) # combine coordinates
    return Pred_grid_xy



# rotation matrix (Azimuth = major axis direction)
@nvtx.annotate("Rot_Mat()", color="purple")
def Rot_Mat(Azimuth, a_max, a_min):
    theta = (Azimuth / 180.0) * cp.pi
    Rot_Mat = cp.dot(
        cp.array([[1 / a_max, 0], [0, 1 / a_min]]),
        cp.array(
            [
                [cp.cos(theta), cp.sin(theta)],
                [-cp.sin(theta), cp.cos(theta)],
            ]
        ),
    )
    return Rot_Mat



# covariance model
@nvtx.annotate("krig_cov()", color="red")
def krig_cov(q, vario):
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
        mat = cp.matmul(q, Rot_Mat(Azimuth, a_max[i], a_min[i]))
        
        #print(mat)
        
        # covariances between measurements
        d = pairwise_distances(mat,mat)
        c = c + covar(vtype[i], d, 1) * cc[i]
    return c

# covariance model
@nvtx.annotate("array_cov()", color="red")
def array_cov(q1, q2, vario):
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
        rot_mat = Rot_Mat(Azimuth, a_max[i], a_min[i])

        mat1 = cp.matmul(q1, rot_mat) 
        mat2 = cp.matmul(q2.reshape(-1,2), rot_mat) 
        # covariances between measurements and unknown
        #d = pairwise_distances(mat1,mat2.T)
        d = cp.sqrt(cp.square(mat1 - mat2).sum(axis=1)) # calculate distances
        #print(d.shape)
        #d = cp.sqrt(cp.square(mat1 - cp.tile(mat2,k)).sum(axis=1)) # calculate distances
        #d = np.asarray(d).reshape(len(d))
        c = c + covar(vtype[i], d, 1) * cc[i] # calculate covariances
    return c

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



######################################

# Simple Kriging Function

######################################


def skrige(Pred_grid, df, xx, yy, data, k, vario, rad):

    """Simple kriging interpolation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :data: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :vario: variogram parameters describing the spatial statistics
    """
    
    Mean_1 = np.average(df[data]) # mean of input data
    Var_1 = np.var(df[data]); # variance of input data 
    
    # preallocate space for mean and variance
    est_SK = np.zeros(shape=len(Pred_grid)) # make zeros array the size of the prediction grid
    var_SK = np.zeros(shape=len(Pred_grid))
    
    # convert dataframe to numpy array for faster matrix operations
    npdata = df[['X','Y','Nbed']].to_numpy()
    
    
    # for each coordinate in the prediction grid
    for z in tqdm(range(0, len(Pred_grid))):
        # gather nearest points within radius
        nearest = nearestNeighborSearch(rad, k, Pred_grid[z], npdata)
        
        # format nearest point bed values matrix
        norm_bed_val = nearest[:,-1]
        norm_bed_val = norm_bed_val.reshape(len(norm_bed_val),1)
        norm_bed_val = norm_bed_val.T
        xy_val = nearest[:, :-1]
       
        # calculate new_k value relative to count of near points within radius
        new_k = len(nearest)
        Kriging_Matrix = np.zeros(shape=((new_k, new_k)))
        Kriging_Matrix = cov(xy_val, xy_val, 0, vario)
            
        r = np.zeros(shape=(new_k))
        k_weights = r
        r = cov(xy_val, np.tile(Pred_grid[z], (new_k, 1)), 1, vario)
        Kriging_Matrix.reshape(((new_k)), ((new_k)))
            
        k_weights = np.dot(np.linalg.pinv(Kriging_Matrix), r)
        est_SK[z] = new_k*Mean_1 + (np.sum(k_weights*(norm_bed_val[:] - Mean_1))) 
        var_SK[z] = (Var_1 - np.sum(k_weights*r))

    return est_SK, var_SK


###########################

# Ordinary Kriging Function

###########################

def okrige(Pred_grid, df, xx, yy, data, k, vario, rad):

    """Ordinary kriging interpolation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :data: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :vario: variogram parameters describing the spatial statistics
    """
    
    Var_1 = np.var(df[data]); # variance of data 
    
    # preallocate space for mean and variance
    est_OK = np.zeros(shape=len(Pred_grid))
    var_OK = np.zeros(shape=len(Pred_grid))
    
    # convert dataframe to numpy matrix for faster operations
    npdata = df[['X','Y','Nbed']].to_numpy()
    
    for z in tqdm(range(0, len(Pred_grid))):
        # find nearest data points
        nearest = nearestNeighborSearch(rad, k, Pred_grid[z], npdata)
        
        # format matrix of nearest bed values
        norm_bed_val = nearest[:,-1]
        norm_bed_val = norm_bed_val.reshape(len(norm_bed_val),1)
        norm_bed_val = norm_bed_val.T
        xy_val = nearest[:,:-1]
        
        # calculate new_k value relative to number of nearby points within radius
        new_k = len(nearest)
        
        # left hand side (covariance between data)
        Kriging_Matrix = np.zeros(shape=((new_k+1, new_k+1)))
        Kriging_Matrix[0:new_k,0:new_k] = cov(xy_val, xy_val, 0, vario)
        Kriging_Matrix[new_k,0:new_k] = 1
        Kriging_Matrix[0:new_k,new_k] = 1
        
        # Set up Right Hand Side (covariance between data and unknown)
        r = np.zeros(shape=(new_k+1))
        k_weights = r
        r[0:new_k] = cov(xy_val, np.tile(Pred_grid[z], (new_k, 1)), 1, vario)
        r[new_k] = 1 # unbiasedness constraint
        Kriging_Matrix.reshape(((new_k+1)), ((new_k+1)))
        
        # Calculate Kriging Weights
        k_weights = np.dot(np.linalg.pinv(Kriging_Matrix), r)

        # get estimates
        est_OK[z] = np.sum(k_weights[0:new_k]*norm_bed_val[:])
        var_OK[z] = Var_1 - np.sum(k_weights[0:new_k]*r[0:new_k])
        
    return est_OK, var_OK




# sequential Gaussian simulation
@nvtx.annotate("sgsim()", color="blue")
def sgsim(Pred_grid, df, xx, yy, data, k, vario, rad):

    """Sequential Gaussian simulation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :df: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :vario: variogram parameters describing the spatial statistics
    """
    
    #print('right version')
    
    # generate random array for simulation order
    xyindex = np.arange(len(Pred_grid))
    random.Random(0).shuffle(xyindex) # random.shuffle(xyindex)

    Var_1 = cp.var(df[data].values); # variance of data 
    
    # preallocate space for simulation
    sgs = cp.zeros(shape=len(Pred_grid))
    
    #for i in tqdm(range(0, len(Pred_grid)), position=0, leave=True):
    for i in tqdm(range(0, 100), position=0, leave=True):
        z = xyindex[i]

        # convert data to numpy array for faster speeds/parsing
        #npdata = df[['X','Y','Nbed']].to_numpy()
        # gather nearby points
        nearest = nearestNeighborSearch(rad, k, Pred_grid[z], df)

        # store X,Y pair values in new array
        xy_val = nearest[["X","Y"]]

        # update K to reflect the amount of K values we got back from quadrant search
        new_k = nearest.shape[0]


        # left hand side (covariance between data)
        Kriging_Matrix = cp.zeros(shape=((new_k+1, new_k+1)))
        Kriging_Matrix[0:new_k,0:new_k] = krig_cov(xy_val.values, vario)
        Kriging_Matrix[new_k,0:new_k] = 1
        Kriging_Matrix[0:new_k,new_k] = 1

        # Set up Right Hand Side (covariance between data and unknown)
        r = cp.zeros(shape=(new_k+1))
        k_weights = r
        r[0:new_k] = array_cov(xy_val.values, cp.tile(Pred_grid[z], new_k), vario) # covariance between simulation grid cell and conditioning data. cp.tile repeats simulation grid cell coordinate entries for covariance calculation
        r[new_k] = 1 # unbiasedness constraint
        Kriging_Matrix.reshape(((new_k+1)), ((new_k+1)))
        
        # Calculate Kriging Weights
        k_weights = cp.dot(cp.linalg.pinv(Kriging_Matrix), r) # lambda = C^-1 * c

        # get estimates
        est = cp.sum(k_weights[0:new_k]*nearest.Nbed.values) # kriging mean
        var = Var_1 - cp.sum(k_weights[0:new_k]*r[0:new_k]) # kriging variance
        
        if (var < 0): # make sure variances are non-negative
            var = 0 
            
        #print(est.shape,var.shape)

        sgs[z] = cp.random.normal(est,cp.sqrt(var),1) # simulate by randomly sampling a value

        # update the conditioning data
        coords = Pred_grid[z:z+1,:]
        df = cudf.concat([df,cudf.DataFrame({"X": [coords[0,0]], "Y": [coords[0,1]], "Nbed": [sgs[z]], "Bed": cp.nan})], sort=False) # add new points by concatenating dataframes 
        
    return sgs


# In[ ]:





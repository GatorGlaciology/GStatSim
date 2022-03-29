#!/usr/bin/env python
# coding: utf-8

# # Sequential Gaussian simulation
# #### Mickey MacKie, Stanford Radio Glaciology
# 
# We perform a sequential Gaussian simulation, a common method for stochastic simulation.

# In[1]:


# load dependencies
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.neighbors import LocalOutlierFactor
import geostatspy.geostats as geostats 

import GlacierStats as gs

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

def main(data_path):



    # ## Load and plot data

    # In[2]:


    df_bed = pd.read_csv(data_path) # download data
    #../Data/Nioghalvfjerds_bed_data.csv
    # remove outliers with LOF method
    clf = LocalOutlierFactor(n_neighbors = 5, contamination = 0.05)
    clf.fit_predict(df_bed)
    lof = clf.negative_outlier_factor_
    df_bed = df_bed[lof >= -1.3]





    # ## Convert to standard Gaussian distribution


    df_bed['Nbed'], tvbed, tnsbed = geostats.nscore(df_bed,'Bed')  # normal score transformation


    # ## Set variogram parameters
    # 
    # These are the variogram model parameters we determined in Variogram_model.ipynb:



    Azimuth = 22.5 # azimuthal direction of major axis
    nug = 0 # nugget
    nstruct = 2 # variogram structures
    vtype = [1,2] # exponential type (1 = spherical, 2 = exponential, 3 = Guassian)
    cc = [0.8,0.2] # contribution for each structure. nugget + cc = 1
    a_max = [40000,50000] # major range for each structure
    a_min = [25000,30000] # minor range for each structure

    vario = [Azimuth, nug, nstruct, vtype, cc, a_max, a_min] # save variogram parameters as a list


    # ## Sequential Gaussian simulation
    # 
    # First we need to define a grid to interpolate:


    # define coordinate grid
    xmin = 420000; xmax = 480000              # range of x values
    ymin = -1090000; ymax = -1030000     # range of y values
    pix = 500  # pixel resolution
    Pred_grid_xy = gs.pred_grid(xmin, xmax, ymin, ymax, pix)


    # In[7]:


    # randomly downsample data to 10% of the original size
    df_samp = df_bed.sample(frac=0.10, replace=False, random_state=1)


    # In[8]:


    k = 50 # number of neighboring data points used to estimate a given point 
    rad = 10000 # 10 km search radius
    sgs = gs.sgsim(Pred_grid_xy, df_samp, 'X', 'Y', 'Nbed', k, vario, rad) # simulate



    # Reverse normal score transformation


    # create dataframe for back transform function
    df_sgs = pd.DataFrame(sgs, columns = ['sgs'])

    # transformation parameters
    vr = tvbed
    vrg = tnsbed
    ltail = 1
    utail = 1
    zmin = -4
    zmax = 4
    ltpar = -1000
    utpar = 1000

    # transformation
    sgs_trans = geostats.backtr(df_sgs,'sgs',vr,vrg,zmin,zmax,ltail,ltpar,utail,utpar)



    # reshape grid
    ylen = (ymax - ymin)/pix
    xlen = (xmax - xmin)/pix
    elevation = np.reshape(sgs_trans, (int(ylen), int(xlen)))

    # multiple realizations
        
    num_sim = 2 # number of realizations
    sgs_mult = np.zeros((num_sim, len(Pred_grid_xy))) # preallocate space for simulations

    # let it rip
    for i in range(0, num_sim):

    # randomly downsample data to 10% of the original size
        df_samp = df_bed.sample(frac=0.10, replace=False, random_state=i) # random_state is optional (this is the seed)
    
        sgs_mult[i,:] = gs.sgsim(Pred_grid_xy, df_samp, 'X', 'Y', 'Nbed', k, vario, rad) # simulate



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str,
        help="File name for csv.",
        default="../Data/Nioghalvfjerds_bed_data.csv")
    args = parser.parse_args()

    print(args.datapath)
    main(data_path = args.datapath)


#!/usr/bin/env python
# coding: utf-8

# In[5]:


### geostatistical tools


# In[3]:


import numpy as np
import numpy.linalg as linalg
import pandas as pd
import sklearn as sklearn
import math
from scipy.spatial import distance_matrix
from tqdm import tqdm
import random
from sklearn.metrics import pairwise_distances


############################

# Grid data

############################

# make array of x,y coordinates based on corners and resolution. This is the grid of values for simulation
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
    
    
# generate coordinates for output of gridded data  
def make_grid(xmin, xmax, ymin, ymax, res):
    cols = np.rint((xmax - xmin)/res); rows = np.rint((ymax - ymin)/res)  # number of rows and columns
    rows = rows.astype(int)
    cols = cols.astype(int)
    x = np.arange(xmin,xmax,res); y = np.arange(ymin,ymax,res) # make arrays
    xx, yy = np.meshgrid(x,y) # make grid

    # shape into array
    x = np.reshape(xx, (int(rows)*int(cols), 1))
    y = np.reshape(yy, (int(rows)*int(cols), 1))

    Pred_grid_xy = np.concatenate((x,y), axis = 1) # combine coordinates
    return Pred_grid_xy, cols, rows


# grid data by averaging the values within each grid cell
def grid_data(df, xx, yy, zz, res):
    
    df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})
    
    xmin = df['X'].min()
    xmax = df['X'].max()
    ymin = df['Y'].min()
    ymax = df['Y'].max()
    grid_coord, cols, rows = make_grid(xmin, xmax, ymin, ymax, res) # make array of grid coordinates
    
    df = df[['X','Y','Z']] # remove any unwanted columns
    np_data = df.to_numpy() # convert to numpy
    np_resize = np.copy(np_data) # copy data
    
    origin = np.array([xmin,ymin])
    resolution = np.array([res,res])
    np_resize[:,:2] = np.rint((np_resize[:,:2]-origin)/resolution) # shift and re-scale the data by subtracting origin and dividing by resolution
    
    grid_sum = np.zeros((rows,cols))
    grid_count = np.copy(grid_sum) # make counter array

    for i in range(np_data.shape[0]):
        xindex = np.int32(np_resize[i,1])
        yindex = np.int32(np_resize[i,0])
        
        if ((xindex >= rows) | (yindex >= cols)):
            continue
            
        grid_sum[xindex,yindex] = np_data[i,2] + grid_sum[xindex,yindex]
        grid_count[xindex,yindex] = 1 + grid_count[xindex,yindex] # add counter
        
    
    np.seterr(invalid='ignore') # ignore erros when dividing by zero (will assign NaN value)
    grid_matrix = np.divide(grid_sum, grid_count) # divide sum by counter to get average within each grid cell
    
    grid_array = np.reshape(grid_matrix,[rows*cols]) # reshape to array
    grid_sum = np.reshape(grid_sum,[rows*cols]) # reshape to array
    grid_count = np.reshape(grid_count,[rows*cols]) # reshape to array
    
    # make dataframe    
    grid_total = np.array([grid_coord[:,0], grid_coord[:,1], grid_sum, grid_count, grid_array])    
    df_grid = pd.DataFrame(grid_total.T, columns = ['X', 'Y', 'Sum', 'Count', 'Z'])  # make dataframe of simulated data
        
    return df_grid, grid_matrix, rows, cols




# make prediction grid for NaN values in gridded data

    
####################################

# Nearest neighbor octant search

####################################

# center data points around grid cell of interest
def center(arrayx, arrayy, centerx, centery):
    centerx = arrayx - centerx
    centery = arrayy - centery
    centered_array = np.array([centerx, centery])
    return centered_array

# calculate distance between array and center coordinates
def distance_calculator(centered_array):
    dist = np.linalg.norm(centered_array, axis=0)
    return dist

# calculate angle between array and center coordinates
def angle_calculator(centered_array):
    angles = np.arctan2(centered_array[0], centered_array[1])
    return angles


def nearestNeighborSearch(rad, count, loc, data2):
    locx = loc[0]
    locy = loc[1]
    
    # wipe coords for re-usability

    data = data2.copy()
    centered_array = center(data['X'].values, data['Y'].values, locx, locy)
    data["dist"] = distance_calculator(centered_array) # compute distance from grid cell of interest
    data["angles"] = angle_calculator(centered_array)
    data = data[data.dist < rad] # delete points outside radius
    data = data.sort_values('dist', ascending = True) # sort array by distances
    bins = [-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi] # break into 8 octants
    data["Oct"] = pd.cut(data.angles, bins = bins, labels = list(range(8))) # octant search
    # number of points to look for in each octant, if not fully divisible by 8, round down
    oct_count = count // 8
    
    smallest = np.ones(shape=(count, 3)) * np.nan

    for i in range(8):
        octant = data[data.Oct == i].iloc[:oct_count][['X','Y','Z']].values # get smallest distance points for each octant
        for j, row in enumerate(octant):
            smallest[i*oct_count+j,:] = row # concatenate octants

    near = smallest[~np.isnan(smallest)].reshape(-1,3) # remove nans

    return near





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




#########################

# Variogram functions

#########################

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


###########################

# Covariance functions

###########################


# covariance matrix (n x n matrix) for covariance between each pair of data points
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
        mat = np.matmul(q, Rot_Mat(Azimuth, a_max[i], a_min[i]))
        
        # covariances between measurements
        d = pairwise_distances(mat,mat)
        c = c + covar(vtype[i], d, 1) * cc[i]
    return c


# n x 1 covariance array for covariance between conditioning data and uknown
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

        mat1 = np.matmul(q1, rot_mat) 
        mat2 = np.matmul(q2.reshape(-1,2), rot_mat) 
        
        # covariances between measurements and unknown
        d = np.sqrt(np.square(mat1 - mat2).sum(axis=1)) # calculate distance
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
def sgsim(Pred_grid, df, xx, yy, zz, k, vario, rad):

    """Sequential Gaussian simulation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :df: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :vario: variogram parameters describing the spatial statistics
    """
    
    # rename header names for consistency with other functions
    df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})
    
    # generate random array for simulation order
    xyindex = np.arange(len(Pred_grid))
    random.shuffle(xyindex)

    Var_1 = np.var(df["Z"].values); # variance of data 
       
    sgs = np.zeros(shape=len(Pred_grid))  # preallocate space for simulation
    
    
    for i in tqdm(range(len(Pred_grid)), position=0, leave=True):
        z = xyindex[i] # get coordinate index
        nearest = nearestNeighborSearch(rad, k, Pred_grid[z], df[['X','Y','Z']])  # gather nearest neighbor points
        norm_bed_val = nearest[:,-1]   # store bed elevation values in new array
        xy_val = nearest[:,:-1]   # store X,Y pair values in new array

        # update K to reflect the amount of K values we got back from quadrant search
        new_k = len(nearest)

        # left hand side (covariance between data)
        Kriging_Matrix = np.zeros(shape=((new_k+1, new_k+1)))
        Kriging_Matrix[0:new_k,0:new_k] = krig_cov(xy_val, vario)
        Kriging_Matrix[new_k,0:new_k] = 1
        Kriging_Matrix[0:new_k,new_k] = 1

        # Set up Right Hand Side (covariance between data and unknown)
        r = np.zeros(shape=(new_k+1))
        k_weights = r
        r[0:new_k] = array_cov(xy_val, np.tile(Pred_grid[z], new_k), vario)
        r[new_k] = 1 # unbiasedness constraint
        Kriging_Matrix.reshape(((new_k+1)), ((new_k+1)))

        k_weights = np.dot(np.linalg.pinv(Kriging_Matrix), r) # Calculate Kriging Weights

        # get estimates
        est = np.sum(k_weights[0:new_k]*norm_bed_val) # kriging mean
        var = Var_1 - np.sum(k_weights[0:new_k]*r[0:new_k]) # kriging variance
        #print(var)

        if (var < 0): # make sure variances are non-negative
            #var = 0 
            var = -var

        #print(var)
        sgs[z] = np.random.normal(est,math.sqrt(var),1) # simulate by randomly sampling a value

        # update the conditioning data
        coords = Pred_grid[z:z+1,:]
        df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 'Z': [sgs[z]]})], sort=False) # add new points by concatenating dataframes 

    return sgs



def sgs_grid(df, xx, yy, zz, res, k, vario, rad):
    
    # rename header names for consistency with other functions
    df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})

    print('Gridding data')
    
    df_grid, grid_matrix, rows, cols = grid_data(df, 'X', 'Y', 'Z', res) # grid data

    df_data = df_grid.dropna()  # get gridded data without NaNs
    df_nan = df_grid[df_grid["Z"].isnull() == True]  # extract coordinates with NaNs
    Pred_grid = df_nan[['X','Y']].to_numpy() # convert pred_grid to numpy for consistency with other functions
    
    print('Simulating')
    sgs = sgsim(Pred_grid, df_data, 'X', 'Y', 'Z', k, vario, rad) # simulate
    
    # add simulated values to conditioning data, and sort based on coordinates
    sim_array = np.array([Pred_grid[:,0],Pred_grid[:,1],sgs])
    sim_array = sim_array.T
    df_sim = pd.DataFrame(sim_array, columns = ['X','Y','Z'])  # make dataframe of simulated data
    df_data = df_data[['X','Y','Z']] # remove unwanted columns

    frames = [df_sim, df_data] # concatenate dataframes
    df_total = pd.concat(frames) 
    
    df_sorted = df_total.sort_values(['X','Y']) # reorder based on coordinates
    
    return df_sorted


# In[ ]:





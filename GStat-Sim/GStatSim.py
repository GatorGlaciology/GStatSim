#!/usr/bin/env python
# coding: utf-8

# In[5]:


### geostatistical tools


# In[3]:


import numpy as np
import numpy.linalg as linalg
import pandas as pd
import sklearn as sklearn
from sklearn.neighbors import KDTree
import math
from scipy.spatial import distance_matrix
from scipy.interpolate import Rbf
from tqdm import tqdm
import random
from sklearn.metrics import pairwise_distances


############################

# Grid data

############################

# make array of x,y coordinates based on corners and resolution. This is the grid of values for simulation
def pred_grid(xmin, xmax, ymin, ymax, pix):
    cols = np.rint((xmax - xmin + pix)/pix); rows = np.rint((ymax - ymin + pix)/pix)  # number of rows and columns
    x = np.arange(xmin,xmax + pix,pix); y = np.arange(ymin,ymax + pix,pix) # make arrays

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
    grid_matrix = np.flipud(grid_matrix) # flip upside down   
    return df_grid, grid_matrix, rows, cols


###################################

# RBF trend estimation

###################################

def rbf_trend(grid_matrix, smooth_radius, res):
    sigma = np.rint(smooth_radius/res)
    ny, nx = grid_matrix.shape
    rbfi = Rbf(np.where(~np.isnan(grid_matrix))[1],np.where(~np.isnan(grid_matrix))[0], grid_matrix[~np.isnan(grid_matrix)],smooth = sigma)

    # evaluate RBF
    yi = np.arange(nx)
    xi = np.arange(ny)
    xi,yi = np.meshgrid(xi, yi)
    trend_rbf = rbfi(xi, yi)   # interpolated values
    return trend_rbf


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
    
    smallest = np.ones(shape=(count, 3)) * np.nan # initialize nan array

    for i in range(8):
        octant = data[data.Oct == i].iloc[:oct_count][['X','Y','Z']].values # get smallest distance points for each octant
        for j, row in enumerate(octant):
            smallest[i*oct_count+j,:] = row # concatenate octants

    near = smallest[~np.isnan(smallest)].reshape(-1,3) # remove nans

    return near


# nearest neighbor search when using cluster_SGS. It finds the nearest neighbor cluster value
def nearestNeighborSearch_cluster(rad, count, loc, data2):
    locx = loc[0]
    locy = loc[1]
    
    # wipe coords for re-usability

    data = data2.copy()
    centered_array = center(data['X'].values, data['Y'].values, locx, locy)
    data["dist"] = distance_calculator(centered_array) # compute distance from grid cell of interest
    data["angles"] = angle_calculator(centered_array)
    data = data[data.dist < rad] # delete points outside radius
    data = data.sort_values('dist', ascending = True) # sort array by distances
    data = data.reset_index() # reset index so that the top index is 0 (so we can extract nearest neighbor K value)
    K = data.K[0] # get nearest neighbor K value
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

    return near, K

# get nearest neighbor secondary data point and coordinates
def nearestNeighborSecondary(loc, data2):
    locx = loc[0]
    locy = loc[1]

    data = data2.copy()
    centered_array = center(data['X'].values, data['Y'].values, locx, locy)
    data["dist"] = distance_calculator(centered_array) # compute distance from grid cell of interest
    data = data.sort_values('dist', ascending = True) # sort array by distances
    data = data.reset_index() # reset index
    nearest_second = data.iloc[0][['X','Y','Z']].values # get coordinates and value of nearest neighbor
    return nearest_second


# find co-located data for co-kriging and co-SGS
def find_colocated(df1, xx1, yy1, zz1, df2, xx2, yy2, zz2):
    
    # rename columns
    df1 = df1.rename(columns = {xx1: "X", yy1: "Y", zz1: "Z"})
    df2 = df2.rename(columns = {xx2: "X", yy2: "Y", zz2: "Z"})
    
    # make KDTree
    secondary_variable_xy = df2[['X','Y']].values
    secondary_variable_tree = KDTree(secondary_variable_xy)

    primary_variable_xy = df1[['X','Y']].values
    nearest_indices = np.zeros(len(primary_variable_xy)) # initialize array of nearest neighbor indices
    
    # query search tree
    for i in range(0,len(primary_variable_xy)):
        nearest_indices[i] = secondary_variable_tree.query(primary_variable_xy[i:i+1,:],
                                                           k=1,return_distance=False)
    
    nearest_indices = np.transpose(nearest_indices)
    secondary_data = df2['Z']
    colocated_secondary_data = secondary_data[nearest_indices]

    df_colocated = pd.DataFrame(np.array(colocated_secondary_data).T, columns = ['colocated'])
    df_colocated.reset_index(drop=True, inplace=True)
    
    return df_colocated


# adaptive partioning recursive implementation
def adaptive_partitioning(df_bed, xmin, xmax, ymin, ymax, i, max_points=100, min_length=25000, max_iter=None):
    """
    Rercursively split clusters until they are all below max_points, but don't go smaller than min_length
    Inputs:
        df_bed - DataFrame with X, Y, and K (cluster id)
        xmin - min x value of this partion
        xmax - max x value of this partion
        ymin - min y value of this partion
        ymax - max y value of this partion
        i - keeps track of total calls to this function
        max_points - all clusters will be "quartered" until points below this
        min_length - minimum side length of sqaures, preference over max_points
        max_iter - maximum iterations if worried about unending recursion
    Outputs:
        df_bed - updated DataFrame with new cluster assigned the next integer
        i - number of iterations
    """
    # optional 'safety' if there is concern about runaway recursion
    if max_iter is not None:
        if i >= max_iter:
            return df_bed, i
    
    dx = xmax - xmin
    dy = ymax - ymin
    
    # >= and <= greedy so we don't miss any points
    xleft = (df_bed.X >= xmin) & (df_bed.X <= xmin+dx/2)
    xright = (df_bed.X <= xmax) & (df_bed.X >= xmin+dx/2)
    ybottom = (df_bed.Y >= ymin) & (df_bed.Y <= ymin+dy/2)
    ytop = (df_bed.Y <= ymax) & (df_bed.Y >= ymin+dy/2)
    
    # index the current cell into 4 quarters
    q1 = df_bed.loc[xleft & ybottom]
    q2 = df_bed.loc[xleft & ytop]
    q3 = df_bed.loc[xright & ytop]
    q4 = df_bed.loc[xright & ybottom]
    
    # for each quarter, qaurter if too many points, else assign K and return
    for q in [q1, q2, q3, q4]:
        if (q.shape[0] > max_points) & (dx/2 > min_length):
            i = i+1
            df_bed, i = adaptive_partitioning(df_bed, q.X.min(), q.X.max(), q.Y.min(), 
                                              q.Y.max(), i, max_points, min_length, max_iter)
        else:
            qcount = df_bed.K.max()
            qcount += 1
            df_bed.loc[q.index, 'K'] = qcount
            
    return df_bed, i


#########################

# Rotation Matrix

#########################


# rotation matrix (Azimuth = major axis direction)
def Rot_Mat(Azimuth, a_maj, a_min):
    theta = (Azimuth / 180.0) * np.pi # convert to radians

    Rot_Mat = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],]),
        np.array([[1 / a_maj, 0], [0, 1 / a_min]]))

    return Rot_Mat


###########################

# Covariance functions

###########################

# covariance function definition. h is effective lag (where range is 1)
def covar(h, sill):
    c = 1 - sill + np.exp(-3 * h) # Exponential covariance function
    return c


# covariance matrix (n x n matrix) for covariance between each pair of data points. rot_mat is a rotation matrix
def krig_cov(q, vario, rot_mat):
    # unpack variogram parameters
    nug = vario[1]
    sill = vario[4] - nug
                                
    c = -nug # nugget effect is made negative because we're calculating covariance instead of variance
    mat = np.matmul(q, rot_mat)

    # covariances between measurements
    h = pairwise_distances(mat,mat) # compute distances 
    c = c + covar(h, sill) # (effective range = 1) compute covariance
    return c


# n x 1 covariance array for covariance between conditioning data and uknown
def array_cov(q1, q2, vario, rot_mat):
    # unpack variogram parameters
    nug = vario[1]
    sill = vario[4] - nug
                            
    c = -nug # nugget effect is made negative because we're calculating covariance instead of variance
    mat1 = np.matmul(q1, rot_mat) 
    mat2 = np.matmul(q2.reshape(-1,2), rot_mat) 
        
    # covariances between measurements and unknown
    h = np.sqrt(np.square(mat1 - mat2).sum(axis=1)) # calculate distance
    c = c + covar(h, sill) # calculate covariances
    return c

    


######################################

# Simple Kriging Function

######################################


def skrige(Pred_grid, df, xx, yy, zz, k, vario, rad):

    """Simple kriging interpolation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :data: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :vario: variogram parameters describing the spatial statistics
    """
    # unpack variogram parameters
    Azimuth = vario[0]
    nug = vario[1]
    a_maj = vario[2]
    a_min = vario[3]
    sill = vario[4]
    rot_mat = Rot_Mat(Azimuth, a_maj, a_min) # rotation matrix for scaling distance based on ranges and anisotropy
    
    # rename header names for consistency with other functions
    df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})
    
    Mean_1 = np.average(df['Z']) # mean of input data
    Var_1 = np.var(df['Z']); # variance of input data 
    
    # preallocate space for mean and variance
    est_SK = np.zeros(shape=len(Pred_grid)) # make zeros array the size of the prediction grid
    var_SK = np.zeros(shape=len(Pred_grid))
    
    
    # for each coordinate in the prediction grid
    for z, predxy in enumerate(tqdm(Pred_grid, position=0, leave=True)):
        test_idx = np.sum(Pred_grid[z]==df[['X', 'Y']].values,axis = 1)
        if np.sum(test_idx==2)==0: # not our hard data
            # gather nearest points within radius
            nearest = nearestNeighborSearch(rad, k, Pred_grid[z], df[['X','Y','Z']])

            # format nearest point bed values matrix
            norm_bed_val = nearest[:,-1]
            norm_bed_val = norm_bed_val.reshape(len(norm_bed_val),1)
            norm_bed_val = norm_bed_val.T
            xy_val = nearest[:, :-1]

            # calculate new_k value relative to count of near points within radius
            new_k = len(nearest)
            Kriging_Matrix = np.zeros(shape=((new_k, new_k)))
            Kriging_Matrix = krig_cov(xy_val, vario, rot_mat)

            r = np.zeros(shape=(new_k))
            k_weights = np.zeros(shape=(new_k))
            r = array_cov(xy_val, np.tile(Pred_grid[z], new_k), vario, rot_mat)
            Kriging_Matrix.reshape(((new_k)), ((new_k)))

            k_weights, res, rank, s = np.linalg.lstsq(Kriging_Matrix, r, rcond = None)
            est_SK[z] = Mean_1 + (np.sum(k_weights*(norm_bed_val[:] - Mean_1))) 
            var_SK[z] = Var_1 - np.sum(k_weights*r)
        else:
            est_SK[z] = df['Z'].values[np.where(test_idx==2)[0][0]]
            var_SK[z] = 0

    return est_SK, var_SK


###########################

# Ordinary Kriging Function

###########################

def okrige(Pred_grid, df, xx, yy, zz, k, vario, rad):

    """Ordinary kriging interpolation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :data: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :vario: variogram parameters describing the spatial statistics
    """
    
    # unpack variogram parameters
    Azimuth = vario[0]
    nug = vario[1]
    a_maj = vario[2]
    a_min = vario[3]
    sill = vario[4]
    rot_mat = Rot_Mat(Azimuth, a_maj, a_min) # rotation matrix for scaling distance based on ranges and anisotropy
    
    # rename header names for consistency with other functions
    df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})
    
    Var_1 = np.var(df['Z']); # variance of data 
    
    # preallocate space for mean and variance
    est_OK = np.zeros(shape=len(Pred_grid))
    var_OK = np.zeros(shape=len(Pred_grid))

    
    for z, predxy in enumerate(tqdm(Pred_grid, position=0, leave=True)):
        test_idx = np.sum(Pred_grid[z]==df[['X', 'Y']].values,axis = 1)
        if np.sum(test_idx==2)==0: # not our hard data
            # find nearest data points
            nearest = nearestNeighborSearch(rad, k, Pred_grid[z], df[['X','Y','Z']])

            # format matrix of nearest bed values
            norm_bed_val = nearest[:,-1]
            local_mean = np.mean(norm_bed_val) # compute the local mean
            norm_bed_val = norm_bed_val.reshape(len(norm_bed_val),1)
            norm_bed_val = norm_bed_val.T
            xy_val = nearest[:,:-1]

            # calculate new_k value relative to number of nearby points within radius
            new_k = len(nearest)

            # left hand side (covariance between data)
            Kriging_Matrix = np.zeros(shape=((new_k+1, new_k+1)))
            Kriging_Matrix[0:new_k,0:new_k] = krig_cov(xy_val, vario, rot_mat)
            Kriging_Matrix[new_k,0:new_k] = 1
            Kriging_Matrix[0:new_k,new_k] = 1

            # Set up Right Hand Side (covariance between data and unknown)
            r = np.zeros(shape=(new_k+1))
            k_weights = np.zeros(shape=(new_k+1))
            r[0:new_k] = array_cov(xy_val, np.tile(Pred_grid[z], new_k), vario, rot_mat)
            r[new_k] = 1 # unbiasedness constraint
            Kriging_Matrix.reshape(((new_k+1)), ((new_k+1)))

            # Calculate Kriging Weights
            k_weights, res, rank, s = np.linalg.lstsq(Kriging_Matrix, r, rcond = None)

            # get estimates
            est_OK[z] = local_mean + np.sum(k_weights[0:new_k]*(norm_bed_val[:] - local_mean))
            var_OK[z] = Var_1 - np.sum(k_weights[0:new_k]*r[0:new_k])
        else:
            est_OK[z] = df['Z'].values[np.where(test_idx==2)[0][0]]
            var_OK[z] = 0
        
    return est_OK, var_OK


# sequential Gaussian simulation using ordinary kriging
def skrige_SGS(Pred_grid, df, xx, yy, zz, k, vario, rad):

    """Sequential Gaussian simulation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :df: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :vario: variogram parameters describing the spatial statistics
    """
    
    # unpack variogram parameters
    Azimuth = vario[0]
    nug = vario[1]
    a_maj = vario[2]
    a_min = vario[3]
    sill = vario[4]
    rot_mat = Rot_Mat(Azimuth, a_maj, a_min) # rotation matrix for scaling distance based on ranges and anisotropy
    
    
    # rename header names for consistency with other functions
    df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})
    
    # generate random array for simulation order
    xyindex = np.arange(len(Pred_grid))
    random.shuffle(xyindex)

    Mean_1 = np.average(df["Z"].values) # mean of input data
    Var_1 = np.var(df["Z"].values) # variance of data 
       
    sgs = np.zeros(shape=len(Pred_grid))  # preallocate space for simulation
    
    
    for idx, predxy in enumerate(tqdm(Pred_grid, position=0, leave=True)):
        z = xyindex[idx] # get coordinate index
        test_idx = np.sum(Pred_grid[z]==df[['X', 'Y']].values,axis = 1)
        if np.sum(test_idx==2)==0: # not our hard data
            nearest = nearestNeighborSearch(rad, k, Pred_grid[z], df[['X','Y','Z']])  # gather nearest neighbor points
            norm_bed_val = nearest[:,-1]   # store bed elevation values in new array
            xy_val = nearest[:,:-1]   # store X,Y pair values in new array

            # update K to reflect the amount of K values we got back from quadrant search
            new_k = len(nearest)

            # left hand side (covariance between data)
            Kriging_Matrix = np.zeros(shape=((new_k, new_k)))
            Kriging_Matrix = krig_cov(xy_val, vario, rot_mat)

            # Set up Right Hand Side (covariance between data and unknown)
            r = np.zeros(shape=(new_k))
            k_weights = np.zeros(shape=(new_k))
            r = array_cov(xy_val, np.tile(Pred_grid[z], new_k), vario, rot_mat)
            Kriging_Matrix.reshape(((new_k)), ((new_k)))

            k_weights, res, rank, s = np.linalg.lstsq(Kriging_Matrix, r, rcond = None) # Calculate Kriging Weights
            #k_weights = np.dot(np.linalg.pinv(Kriging_Matrix), r) 

            # get estimates
            est = Mean_1 + np.sum(k_weights*(norm_bed_val - Mean_1)) # simple kriging mean
            var = Var_1 - np.sum(k_weights*r) # simple kriging variance
            var = np.absolute(var) # make sure variances are non-negative

            sgs[z] = np.random.normal(est,math.sqrt(var),1) # simulate by randomly sampling a value
        else:
            sgs[z] = df['Z'].values[np.where(test_idx==2)[0][0]]

        # update the conditioning data
        coords = Pred_grid[z:z+1,:]
        df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 'Z': [sgs[z]]})], sort=False) # add new points by concatenating dataframes 

    return sgs


# sequential Gaussian simulation using ordinary kriging
def okrige_SGS(Pred_grid, df, xx, yy, zz, k, vario, rad):

    """Sequential Gaussian simulation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :df: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :vario: variogram parameters describing the spatial statistics
    """
    
    # unpack variogram parameters
    Azimuth = vario[0]
    nug = vario[1]
    a_maj = vario[2]
    a_min = vario[3]
    sill = vario[4]
    rot_mat = Rot_Mat(Azimuth, a_maj, a_min) # rotation matrix for scaling distance based on ranges and anisotropy
    
    # rename header names for consistency with other functions
    df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})
    
    # generate random array for simulation order
    xyindex = np.arange(len(Pred_grid))
    random.shuffle(xyindex)
    Var_1 = np.var(df["Z"].values) # variance of data 
       
    sgs = np.zeros(shape=len(Pred_grid))  # preallocate space for simulation
    
    
    for idx, predxy in enumerate(tqdm(Pred_grid, position=0, leave=True)):
        z = xyindex[idx] # get coordinate index
        test_idx = np.sum(Pred_grid[z]==df[['X', 'Y']].values,axis = 1)
        if np.sum(test_idx==2)==0: # not our hard data
            nearest = nearestNeighborSearch(rad, k, Pred_grid[z], df[['X','Y','Z']])  # gather nearest neighbor points
            norm_bed_val = nearest[:,-1]   # store bed elevation values in new array
            xy_val = nearest[:,:-1]   # store X,Y pair values in new array
            local_mean = np.mean(norm_bed_val) # compute the local mean

            # update K to reflect the amount of K values we got back from quadrant search
            new_k = len(nearest)

            # left hand side (covariance between data)
            Kriging_Matrix = np.zeros(shape=((new_k+1, new_k+1)))
            Kriging_Matrix[0:new_k,0:new_k] = krig_cov(xy_val, vario, rot_mat)
            Kriging_Matrix[new_k,0:new_k] = 1
            Kriging_Matrix[0:new_k,new_k] = 1

            # Set up Right Hand Side (covariance between data and unknown)
            r = np.zeros(shape=(new_k+1))
            k_weights = np.zeros(shape=(new_k+1))
            r[0:new_k] = array_cov(xy_val, np.tile(Pred_grid[z], new_k), vario, rot_mat)
            r[new_k] = 1 # unbiasedness constraint
            Kriging_Matrix.reshape(((new_k+1)), ((new_k+1)))

            k_weights, res, rank, s = np.linalg.lstsq(Kriging_Matrix, r, rcond = None) # Calculate Kriging Weights
            #k_weights = np.dot(np.linalg.pinv(Kriging_Matrix), r) # Calculate Kriging Weights

            # get estimates
            est = local_mean + np.sum(k_weights[0:new_k]*(norm_bed_val - local_mean)) # kriging mean
            var = Var_1 - np.sum(k_weights[0:new_k]*r[0:new_k]) # kriging variance
            var = np.absolute(var) # make sure variances are non-negative

            sgs[z] = np.random.normal(est,math.sqrt(var),1) # simulate by randomly sampling a value
        else:
            sgs[z] = df['Z'].values[np.where(test_idx==2)[0][0]] 

        # update the conditioning data
        coords = Pred_grid[z:z+1,:]
        df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 'Z': [sgs[z]]})], sort=False) # add new points by concatenating dataframes 

    return sgs

# sgsim with multiple clusters
def cluster_SGS(Pred_grid, df, xx, yy, zz, kk, k, df_gamma, rad):

    """Sequential Gaussian simulation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :zz: y variable
    :kk: k-means cluster number
    :df: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :df_gamma: dataframe of variogram parameters describing the spatial statistics for each cluster
    """
   
    
    # rename header names for consistency with other functions
    df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z", kk: "K"})
    
    # generate random array for simulation order
    xyindex = np.arange(len(Pred_grid))
    random.shuffle(xyindex)

    #for i in range(len(df_gamma)
    Mean_1 = np.average(df["Z"].values) # mean of input data
    Var_1 = np.var(df["Z"].values); # variance of data 
       
    sgs = np.zeros(shape=len(Pred_grid))  # preallocate space for simulation
    
    
    for idx, predxy in enumerate(tqdm(Pred_grid, position=0, leave=True)):
        z = xyindex[idx] # get coordinate index
        test_idx = np.sum(Pred_grid[z]==df[['X', 'Y']].values,axis = 1)
        if np.sum(test_idx==2)==0: # not our hard data
            nearest, K = nearestNeighborSearch_cluster(rad, k, Pred_grid[z], df[['X','Y','Z','K']])  # gather nearest neighbor points and K cluster value
            vario = df_gamma.Variogram[K] # define variogram parameters using cluster value
            norm_bed_val = nearest[:,-1]   # store bed elevation values in new array
            xy_val = nearest[:,:-1]   # store X,Y pair values in new array

            # unpack variogram parameters
            Azimuth = vario[0]
            nug = vario[1]
            a_maj = vario[2]
            a_min = vario[3]
            sill = vario[4]
            rot_mat = Rot_Mat(Azimuth, a_maj, a_min) # rotation matrix for scaling distance based on ranges and anisotropy

            # update K to reflect the amount of K values we got back from quadrant search
            new_k = len(nearest)

            # left hand side (covariance between data)
            Kriging_Matrix = np.zeros(shape=((new_k, new_k)))
            Kriging_Matrix[0:new_k,0:new_k] = krig_cov(xy_val, vario, rot_mat) 

            # Set up Right Hand Side (covariance between data and unknown)
            r = np.zeros(shape=(new_k))
            k_weights = np.zeros(shape=(new_k))
            r = array_cov(xy_val, np.tile(Pred_grid[z], new_k), vario, rot_mat)
            Kriging_Matrix.reshape(((new_k)), ((new_k)))

            k_weights, res, rank, s = np.linalg.lstsq(Kriging_Matrix, r, rcond = None) # Calculate Kriging Weights

            # get estimates
            est = Mean_1 + np.sum(k_weights*(norm_bed_val - Mean_1)) # simple kriging mean
            var = Var_1 - np.sum(k_weights*r) # simple kriging variance
            var = np.absolute(var) # make sure variances are non-negative

            #print(var)
            sgs[z] = np.random.normal(est,math.sqrt(var),1) # simulate by randomly sampling a value
        else:
            sgs[z] = df['Z'].values[np.where(test_idx==2)[0][0]] 
            
        # update the conditioning data
        coords = Pred_grid[z:z+1,:]
        df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 'Z': [sgs[z]], 'K': [K]})], sort=False) # add new points and K value by concatenating dataframes 

    return sgs


###########################

# Multivariate

###########################

# perform simple collocated cokriging with MM1
def cokrige_mm1(Pred_grid, df1, xx1, yy1, zz1, df2, xx2, yy2, zz2, k, vario, rad, corrcoef):

    # unpack variogram parameters for rotation matrix
    Azimuth = vario[0]
    a_maj = vario[2]
    a_min = vario[3]
    rot_mat = Rot_Mat(Azimuth, a_maj, a_min) # rotation matrix for scaling distance based on ranges and anisotropy
    
    # rename header names for consistency with other functions
    df1 = df1.rename(columns = {xx1: "X", yy1: "Y", zz1: "Z"})
    df2 = df2.rename(columns = {xx2: "X", yy2: "Y", zz2: "Z"})
    
    Mean_1 = np.average(df1['Z']) # mean of primary data
    Var_1 = np.var(df1['Z']); # variance of primary data 
    Mean_2 = np.average(df2['Z']) # mean of secondary data
    Var_2 = np.var(df2['Z']); # variance of secondary data 
    
    # preallocate space for mean and variance
    est_cokrige = np.zeros(shape=len(Pred_grid)) # make zeros array the size of the prediction grid
    var_cokrige = np.zeros(shape=len(Pred_grid))
    
    
    # for each coordinate in the prediction grid
    for z, predxy in enumerate(tqdm(Pred_grid, position=0, leave=True)):
        test_idx = np.sum(Pred_grid[z]==df1[['X', 'Y']].values,axis = 1)
        if np.sum(test_idx==2)==0: # not our hard data
            # gather nearest primary data points within radius
            nearest = nearestNeighborSearch(rad, k, Pred_grid[z], df1[['X','Y','Z']])

            # get nearest neighbor secondary data point
            nearest_second = nearestNeighborSecondary(Pred_grid[z], df2[['X','Y','Z']])

            # format nearest point bed values matrix
            norm_bed_val = nearest[:,-1] # values of nearest neighbor points
            norm_bed_val = norm_bed_val.reshape(len(norm_bed_val),1)
            norm_bed_val = norm_bed_val.T
            norm_bed_val = np.append(norm_bed_val, [nearest_second[-1]]) # append secondary data value
            xy_val = nearest[:, :-1] # coordinates of nearest neighbor points
            xy_second = nearest_second[:-1] # secondary data coordinates
            xy_val = np.append(xy_val, [xy_second], axis = 0) # append coordinates of secondary data

            # set up covariance matrix
            new_k = len(nearest)
            Kriging_Matrix = np.zeros(shape=((new_k + 1, new_k + 1)))
            Kriging_Matrix[0:new_k+1, 0:new_k+1] = krig_cov(xy_val, vario, rot_mat) # covariance within primary data

            # get covariance between data and unknown grid cell
            r = np.zeros(shape=(new_k + 1))
            k_weights = np.zeros(shape=(new_k + 1))
            r[0:new_k+1] = array_cov(xy_val, np.tile(Pred_grid[z], new_k + 1), vario, rot_mat)
            r[new_k] = r[new_k] * corrcoef # correlation between primary and nearest neighbor (zero lag) secondary data

            # update covariance matrix with secondary info (gamma2 = rho12 * gamma1)
            Kriging_Matrix[new_k, 0 : new_k+1] = Kriging_Matrix[new_k, 0 : new_k+1] * corrcoef
            Kriging_Matrix[0 : new_k+1, new_k] = Kriging_Matrix[0 : new_k+1, new_k] * corrcoef
            Kriging_Matrix[new_k, new_k] = 1
            Kriging_Matrix.reshape(((new_k + 1)), ((new_k + 1)))

            # solve kriging system
            k_weights, res, rank, s = np.linalg.lstsq(Kriging_Matrix, r, rcond = None) # get weights
            part1 = Mean_1 + np.sum(k_weights[0:new_k]*(norm_bed_val[0:new_k] - Mean_1)/np.sqrt(Var_1))
            part2 = k_weights[new_k] * (nearest_second[-1] - Mean_2)/np.sqrt(Var_2)
            est_cokrige[z] = part1 + part2 # compute mean
            var_cokrige[z] = 1 - np.sum(k_weights*r) # compute variance
        else:
            est_cokrige[z] = df1['Z'].values[np.where(test_idx==2)[0][0]]
            var_cokrige[z] = 0

    return est_cokrige, var_cokrige
    
    
# perform cosimulation with MM1
def cosim_mm1(Pred_grid, df1, xx1, yy1, zz1, df2, xx2, yy2, zz2, k, vario, rad, corrcoef):

    # unpack variogram parameters
    Azimuth = vario[0]
    a_maj = vario[2]
    a_min = vario[3]
    rot_mat = Rot_Mat(Azimuth, a_maj, a_min) # rotation matrix for scaling distance based on ranges and anisotropy
    
    # rename header names for consistency with other functions
    df1 = df1.rename(columns = {xx1: "X", yy1: "Y", zz1: "Z"})
    df2 = df2.rename(columns = {xx2: "X", yy2: "Y", zz2: "Z"})
    
    # generate random array for simulation order
    xyindex = np.arange(len(Pred_grid))
    random.shuffle(xyindex)
    
    Mean_1 = np.average(df1['Z']) # mean of primary data
    Var_1 = np.var(df1['Z']); # variance of primary data 
    Mean_2 = np.average(df2['Z']) # mean of secondary data
    Var_2 = np.var(df2['Z']); # variance of secondary data 
    
    # preallocate space for mean and variance
    est_cokrige = np.zeros(shape=len(Pred_grid)) # make zeros array the size of the prediction grid
    var_cokrige = np.zeros(shape=len(Pred_grid))
    
    cosim = np.zeros(shape=len(Pred_grid))  # preallocate space for simulation
    
    # for each coordinate in the prediction grid
    for idx, predxy in enumerate(tqdm(Pred_grid, position=0, leave=True)):
        z = xyindex[idx] # get coordinate index
        test_idx = np.sum(Pred_grid[z]==df1[['X', 'Y']].values,axis = 1)
        if np.sum(test_idx==2)==0: # not our hard data
            # gather nearest primary data points within radius
            nearest = nearestNeighborSearch(rad, k, Pred_grid[z], df1[['X','Y','Z']])

            # get nearest neighbor secondary data point
            nearest_second = nearestNeighborSecondary(Pred_grid[z], df2[['X','Y','Z']])

            # format nearest point bed values matrix
            norm_bed_val = nearest[:,-1] # values of nearest neighbor points
            norm_bed_val = norm_bed_val.reshape(len(norm_bed_val),1)
            norm_bed_val = norm_bed_val.T
            norm_bed_val = np.append(norm_bed_val, [nearest_second[-1]]) # append secondary data value
            xy_val = nearest[:, :-1] # coordinates of nearest neighbor points
            xy_second = nearest_second[:-1] # secondary data coordinates
            xy_val = np.append(xy_val, [xy_second], axis = 0) # append coordinates of secondary data

            # set up covariance matrix
            new_k = len(nearest)
            Kriging_Matrix = np.zeros(shape=((new_k + 1, new_k + 1)))
            Kriging_Matrix[0:new_k+1, 0:new_k+1] = krig_cov(xy_val, vario, rot_mat) # covariance within primary data

            # get covariance between data and unknown grid cell
            r = np.zeros(shape=(new_k + 1))
            k_weights = np.zeros(shape=(new_k + 1))
            r[0:new_k+1] = array_cov(xy_val, np.tile(Pred_grid[z], new_k + 1), vario, rot_mat)
            r[new_k] = r[new_k] * corrcoef # correlation between primary and nearest neighbor (zero lag) secondary data

            # update covariance matrix with secondary info (gamma2 = rho12 * gamma1)
            Kriging_Matrix[new_k, 0 : new_k+1] = Kriging_Matrix[new_k, 0 : new_k+1] * corrcoef
            Kriging_Matrix[0 : new_k+1, new_k] = Kriging_Matrix[0 : new_k+1, new_k] * corrcoef
            Kriging_Matrix[new_k, new_k] = 1
            Kriging_Matrix.reshape(((new_k + 1)), ((new_k + 1)))

            # solve kriging system
            k_weights, res, rank, s = np.linalg.lstsq(Kriging_Matrix, r, rcond = None) # get weights
            part1 = Mean_1 + np.sum(k_weights[0:new_k]*(norm_bed_val[0:new_k] - Mean_1)/np.sqrt(Var_1))
            part2 = k_weights[new_k] * (nearest_second[-1] - Mean_2)/np.sqrt(Var_2)
            est_cokrige = part1 + part2 # compute mean
            var_cokrige = 1 - np.sum(k_weights*r) # compute variance
            var_cokrige = np.absolute(var_cokrige) # make sure variances are non-negative

            cosim[z] = np.random.normal(est_cokrige,math.sqrt(var_cokrige),1) # simulate by randomly sampling a value
        else:
            cosim[z] = df1['Z'].values[np.where(test_idx==2)[0][0]] 

        # update the conditioning data
        coords = Pred_grid[z:z+1,:]
        df1 = pd.concat([df1,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 'Z': [cosim[z]]})], sort=False) # add new points by concatenating dataframes 
        
    return cosim

# In[ ]:





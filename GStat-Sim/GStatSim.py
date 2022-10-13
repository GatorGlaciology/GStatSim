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


###############################################################################

# Grid data

############################

class Gridding:

    # make array of x,y coordinates based on corners and resolution. 
    # This is the grid of values for simulation
    def prediction_grid(XMIN, XMAX, YMIN, YMAX, RES):
        cols = np.rint((XMAX - XMIN + RES)/RES)
        rows = np.rint((YMAX - YMIN + RES)/RES)  # number of rows and columns
        x = np.arange(XMIN,XMAX + RES,RES); y = np.arange(YMIN,YMAX + RES,RES) 
        xx, yy = np.meshgrid(x,y) # make grid
        yy = np.flip(yy) # flip upside down
        x = np.reshape(xx, (int(rows)*int(cols), 1)) # shape into array
        y = np.reshape(yy, (int(rows)*int(cols), 1))
        prediction_grid_xy = np.concatenate((x,y), axis = 1) # combine coordinates
        
        return prediction_grid_xy


    # generate coordinates for output of gridded data  
    def make_grid(XMIN, XMAX, YMIN, YMAX, RES):
        # number of rows and columns
        cols = np.rint((XMAX - XMIN)/RES) 
        rows = np.rint((YMAX - YMIN)/RES)  
        rows = rows.astype(int)
        cols = cols.astype(int)
        x = np.arange(XMIN,XMAX,RES); y = np.arange(YMIN,YMAX,RES) # make arrays
        xx, yy = np.meshgrid(x,y) # make grid
        x = np.reshape(xx, (int(rows)*int(cols), 1)) # shape into array
        y = np.reshape(yy, (int(rows)*int(cols), 1))
        prediction_grid_xy = np.concatenate((x,y), axis = 1) # combine coordinates
        
        return prediction_grid_xy, cols, rows
    
    
    # grid data by averaging the values within each grid cell
    def grid_data(df, xx, yy, zz, RES):
        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})

        XMIN = df['X'].min()
        XMAX = df['X'].max()
        YMIN = df['Y'].min()
        YMAX = df['Y'].max()
        
        # make array of grid coordinates
        grid_coord, cols, rows = Gridding.make_grid(XMIN, XMAX, YMIN, YMAX, RES) 

        df = df[['X','Y','Z']] # remove any unwanted columns
        np_data = df.to_numpy() # convert to numpy
        np_resize = np.copy(np_data) # copy data
        origin = np.array([XMIN,YMIN])
        resolution = np.array([RES,RES])
        
        # shift and re-scale the data by subtracting origin and dividing by resolution
        np_resize[:,:2] = np.rint((np_resize[:,:2]-origin)/resolution) 

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
        grid_total = np.array([grid_coord[:,0], grid_coord[:,1], 
                               grid_sum, grid_count, grid_array])    
        df_grid = pd.DataFrame(grid_total.T, 
                               columns = ['X', 'Y', 'Sum', 'Count', 'Z'])  # make dataframe of simulated data
        grid_matrix = np.flipud(grid_matrix) # flip upside down   
        
        return df_grid, grid_matrix, rows, cols


###################################

# RBF trend estimation

###################################

def rbf_trend(grid_matrix, smooth_factor, RES):
    sigma = np.rint(smooth_factor/RES)
    ny, nx = grid_matrix.shape
    rbfi = Rbf(np.where(~np.isnan(grid_matrix))[1],
               np.where(~np.isnan(grid_matrix))[0], 
               grid_matrix[~np.isnan(grid_matrix)],smooth = sigma)

    # evaluate RBF
    yi = np.arange(nx)
    xi = np.arange(ny)
    xi,yi = np.meshgrid(xi, yi)
    trend_rbf = rbfi(xi, yi)   # interpolated values
    
    return trend_rbf


####################################

# Nearest neighbor octant search

####################################

class NearestNeighbor:

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

    def nearest_neighbor_search(RADIUS, NUM_POINTS, loc, data2):
        locx = loc[0]
        locy = loc[1]
        data = data2.copy()
        centered_array = NearestNeighbor.center(data['X'].values, data['Y'].values, 
                                locx, locy)
        data["dist"] = NearestNeighbor.distance_calculator(centered_array) # compute distance from grid cell of interest
        data["angles"] = NearestNeighbor.angle_calculator(centered_array)
        data = data[data.dist < RADIUS] # delete points outside radius
        data = data.sort_values('dist', ascending = True) # sort array by distances
        bins = [-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0, 
                math.pi/4, math.pi/2, 3*math.pi/4, math.pi] # break into 8 octants
        data["Oct"] = pd.cut(data.angles, bins = bins, labels = list(range(8))) # octant search
        # number of points to look for in each octant, if not fully divisible by 8, round down
        oct_count = NUM_POINTS // 8
        smallest = np.ones(shape=(NUM_POINTS, 3)) * np.nan # initialize nan array

        for i in range(8):
            octant = data[data.Oct == i].iloc[:oct_count][['X','Y','Z']].values # get smallest distance points for each octant
            for j, row in enumerate(octant):
                smallest[i*oct_count+j,:] = row # concatenate octants
        near = smallest[~np.isnan(smallest)].reshape(-1,3) # remove nans
        
        return near


    # nearest neighbor search when using cluster_SGS. It finds the nearest neighbor cluster value
    def nearest_neighbor_search_cluster(RADIUS, NUM_POINTS, loc, data2):
        locx = loc[0]
        locy = loc[1]
        data = data2.copy()
        centered_array = NearestNeighbor.center(data['X'].values, data['Y'].values, 
                                locx, locy)
        data["dist"] = NearestNeighbor.distance_calculator(centered_array) # compute distance from grid cell of interest
        data["angles"] = NearestNeighbor.angle_calculator(centered_array)
        data = data[data.dist < RADIUS] # delete points outside radius
        data = data.sort_values('dist', ascending = True) # sort array by distances
        data = data.reset_index() # reset index so that the top index is 0 (so we can extract nearest neighbor K value)
        cluster_number = data.K[0] # get nearest neighbor cluster value
        bins = [-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0, 
                math.pi/4, math.pi/2, 3*math.pi/4, math.pi] # break into 8 octants
        data["Oct"] = pd.cut(data.angles, bins = bins, labels = list(range(8))) # octant search
        # number of points to look for in each octant, if not fully divisible by 8, round down
        oct_count = NUM_POINTS // 8
        smallest = np.ones(shape=(NUM_POINTS, 3)) * np.nan

        for i in range(8):
            octant = data[data.Oct == i].iloc[:oct_count][['X','Y','Z']].values # get smallest distance points for each octant
            for j, row in enumerate(octant):
                smallest[i*oct_count+j,:] = row # concatenate octants
        near = smallest[~np.isnan(smallest)].reshape(-1,3) # remove nans
        
        return near, cluster_number

    # get nearest neighbor secondary data point and coordinates
    def nearest_neighbor_secondary(loc, data2):
        locx = loc[0]
        locy = loc[1]
        data = data2.copy()
        centered_array = NearestNeighbor.center(data['X'].values, data['Y'].values, 
                                locx, locy)
        data["dist"] = NearestNeighbor.distance_calculator(centered_array) # compute distance from grid cell of interest
        data = data.sort_values('dist', ascending = True) # sort array by distances
        data = data.reset_index() # reset index
        nearest_second = data.iloc[0][['X','Y','Z']].values # get coordinates and value of nearest neighbor
        
        return nearest_second


    # find co-located data for co-kriging and co-SGS
    def find_colocated(df1, xx1, yy1, zz1, df2, xx2, yy2, zz2):     
        df1 = df1.rename(columns = {xx1: "X", yy1: "Y", zz1: "Z"}) # rename columns
        df2 = df2.rename(columns = {xx2: "X", yy2: "Y", zz2: "Z"}) 
        secondary_variable_xy = df2[['X','Y']].values
        secondary_variable_tree = KDTree(secondary_variable_xy) # make KDTree
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
def adaptive_partitioning(df_data, XMIN, XMAX, YMIN, YMAX, i, MAX_POINTS, MIN_LENGTH, MAX_ITER=None):
    """
    Rercursively split clusters until they are all below max_points, but don't go smaller than min_length
    Inputs:
        df_data - DataFrame with X, Y, and K (cluster id)
        XMIN - min x value of this partion
        XMAX - max x value of this partion
        YMIN - min y value of this partion
        YMAX - max y value of this partion
        i - keeps track of total calls to this function
        max_points - all clusters will be "quartered" until points below this
        min_length - minimum side length of sqaures, preference over max_points
        max_iter - maximum iterations if worried about unending recursion
    Outputs:
        df_data - updated DataFrame with new cluster assigned the next integer
        i - number of iterations
    """
    # optional 'safety' if there is concern about runaway recursion
    if MAX_ITER is not None:
        if i >= MAX_ITER:
            return df_data, i
    
    dx = XMAX - XMIN
    dy = YMAX - YMIN
    
    # >= and <= greedy so we don't miss any points
    xleft = (df_data.X >= XMIN) & (df_data.X <= XMIN+dx/2)
    xright = (df_data.X <= XMAX) & (df_data.X >= XMIN+dx/2)
    ybottom = (df_data.Y >= YMIN) & (df_data.Y <= YMIN+dy/2)
    ytop = (df_data.Y <= YMAX) & (df_data.Y >= YMIN+dy/2)
    
    # index the current cell into 4 quarters
    q1 = df_data.loc[xleft & ybottom]
    q2 = df_data.loc[xleft & ytop]
    q3 = df_data.loc[xright & ytop]
    q4 = df_data.loc[xright & ybottom]
    
    # for each quarter, qaurter if too many points, else assign K and return
    for q in [q1, q2, q3, q4]:
        if (q.shape[0] > MAX_POINTS) & (dx/2 > MIN_LENGTH):
            i = i+1
            df_data, i = adaptive_partitioning(df_data, q.X.min(), 
                                               q.X.max(), q.Y.min(), q.Y.max(), i, 
                                               MAX_POINTS, MIN_LENGTH, MAX_ITER)
        else:
            qcount = df_data.K.max()
            qcount += 1
            df_data.loc[q.index, 'K'] = qcount
            
            # make clusters zero indexed
            #df_data.K = df_data.K.astype(int) - 1
            
    return df_data, i


#########################

# Rotation Matrix

#########################


# make rotation matrix (AZIMUTH = major axis direction)
def make_rotation_matrix(AZIMUTH, MAJOR_RANGE, MINOR_RANGE):
    theta = (AZIMUTH / 180.0) * np.pi # convert to radians
    
    rotation_matrix = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],]),
        np.array([[1 / MAJOR_RANGE, 0], [0, 1 / MINOR_RANGE]]))
    
    return rotation_matrix


###########################

# Covariance functions

###########################

class Covariance:

    # covariance function definition. h is effective lag (where range is 1)
    def covar(effective_lag, SILL):
        c = 1 - SILL + np.exp(-3 * effective_lag) # Exponential covariance function

        return c


    # covariance matrix (n x n matrix) for covariance between each pair of coordinates.
    def make_covariance_matrix(coord, vario, rotation_matrix):
        # unpack variogram parameters
        NUG = vario[1]
        SILL = vario[4] - NUG                                
        c = -NUG # nugget effect is made negative because we're calculating covariance instead of variance
        mat = np.matmul(coord, rotation_matrix)
        effective_lag = pairwise_distances(mat,mat) # compute distances 
        covariance_matrix = c + Covariance.covar(effective_lag, SILL) # (effective range = 1) compute covariance

        return covariance_matrix


    # n x 1 covariance array for covariance between conditioning data and uknown
    def make_covariance_array(coord1, coord2, vario, rotation_matrix):
        # unpack variogram parameters
        NUG = vario[1]
        SILL = vario[4] - NUG                          
        c = -NUG # nugget effect is made negative because we're calculating covariance instead of variance
        mat1 = np.matmul(coord1, rotation_matrix) 
        mat2 = np.matmul(coord2.reshape(-1,2), rotation_matrix) 
        effective_lag = np.sqrt(np.square(mat1 - mat2).sum(axis=1)) # calculate distance
        covariance_array = c + Covariance.covar(effective_lag, SILL) # calculate covariances

        return covariance_array
    # 

# Simple Kriging Function

######################################
class Interpolation: 

    def skrige(prediction_grid, df, xx, yy, zz, NUM_POINTS, vario, RADIUS):

        """Simple kriging interpolation
        :param prediction_grid: x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
        :df: data frame of input data
        :xx: column name for x coordinates of input data frame
        :yy: column name for y coordinates of input data frame
        :data: column name for the data variable (in this case, bed elevation) of the input data frame
        :NUM_POINTS: the number of conditioning points to search for
        :vario: variogram parameters describing the spatial statistics
        :RADIUS: search radius
        """
        # unpack variogram parameters
        AZIMUTH = vario[0]
        MAJOR_RANGE = vario[2]
        MINOR_RANGE = vario[3]
        rotation_matrix = make_rotation_matrix(AZIMUTH, MAJOR_RANGE, MINOR_RANGE) # rotation matrix for scaling distance based on ranges and anisotropy

        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"}) # rename header names for consistency with other functions   
        MEAN_1 = np.average(df['Z']) # mean of input data
        VAR_1 = np.var(df['Z']); # variance of input data 
        est_sk = np.zeros(shape=len(prediction_grid)) # preallocate space for mean and variance
        var_sk = np.zeros(shape=len(prediction_grid))

        # for each coordinate in the prediction grid
        for z, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0: # not our hard data
                
                # gather nearest points within radius
                nearest = NearestNeighbor.nearest_neighbor_search(RADIUS, NUM_POINTS, 
                                                  prediction_grid[z], df[['X','Y','Z']])
                norm_data_val = nearest[:,-1] # format nearest point data values matrix
                norm_data_val = norm_data_val.reshape(len(norm_data_val),1)
                norm_data_val = norm_data_val.T
                xy_val = nearest[:, :-1]   
                new_num_pts = len(nearest) 

                covariance_matrix = np.zeros(shape=((new_num_pts, new_num_pts)))
                covariance_matrix = Covariance.make_covariance_matrix(xy_val, 
                                                           vario, rotation_matrix)

                # covariance between data and uknown
                covariance_array = np.zeros(shape=(new_num_pts))
                k_weights = np.zeros(shape=(new_num_pts))
                covariance_array = Covariance.make_covariance_array(xy_val, 
                                                         np.tile(prediction_grid[z], new_num_pts), 
                                                         vario, rotation_matrix)
                
                # covariance between data points
                covariance_matrix.reshape(((new_num_pts)), ((new_num_pts)))
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array, rcond = None)
                
                # to avoid underestimating variance, calculate weights for variance estimation separately
                vario2 = [vario[0], vario[1], vario[2], vario[3], VAR_1]
                covariance_array2 = Covariance.make_covariance_array(xy_val, 
                                                         np.tile(prediction_grid[z], new_num_pts), 
                                                         vario2, rotation_matrix)
                k_weights2, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array2, rcond = None)

                est_sk[z] = MEAN_1 + (np.sum(k_weights*(norm_data_val[:] - MEAN_1))) 
                var_sk[z] = VAR_1 - np.sum(k_weights2*covariance_array2)
            else:
                est_sk[z] = df['Z'].values[np.where(test_idx==2)[0][0]]
                var_sk[z] = 0

        return est_sk, var_sk


    ###########################

    # Ordinary Kriging Function

    ###########################

    def okrige(prediction_grid, df, xx, yy, zz, NUM_POINTS, vario, RADIUS):

        """Ordinary kriging interpolation
        :param prediction_grid: x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
        :df: data frame of input data
        :xx: column name for x coordinates of input data frame
        :yy: column name for y coordinates of input data frame
        :data: column name for the data variable (in this case, bed elevation) of the input data frame
        :NUM_POINTS: the number of conditioning points to search for
        :vario: variogram parameters describing the spatial statistics
        """

        # unpack variogram parameters
        AZIMUTH = vario[0]
        MAJOR_RANGE = vario[2]
        MINOR_RANGE = vario[3]
        rotation_matrix = make_rotation_matrix(AZIMUTH, MAJOR_RANGE, MINOR_RANGE) # rotation matrix for scaling distance based on ranges and anisotropy

        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"}) # rename header names for consistency with other functions    
        VAR_1 = np.var(df['Z']); # variance of data     
        est_ok = np.zeros(shape=len(prediction_grid)) # preallocate space for mean and variance
        var_ok = np.zeros(shape=len(prediction_grid))

        for z, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0: # not our hard data
                
                # find nearest data points
                nearest = NearestNeighbor.nearest_neighbor_search(RADIUS, NUM_POINTS, 
                                                  prediction_grid[z], df[['X','Y','Z']])    
                norm_data_val = nearest[:,-1] # format matrix of nearest data values
                local_mean = np.mean(norm_data_val) # compute the local mean
                norm_data_val = norm_data_val.reshape(len(norm_data_val),1)
                norm_data_val = norm_data_val.T
                xy_val = nearest[:,:-1]
                new_num_pts = len(nearest) 

                # left hand side (covariance between data)
                covariance_matrix = np.zeros(shape=((new_num_pts+1, new_num_pts+1))) 
                covariance_matrix[0:new_num_pts,0:new_num_pts] = Covariance.make_covariance_matrix(xy_val, 
                                                                                        vario, rotation_matrix)
                covariance_matrix[new_num_pts,0:new_num_pts] = 1
                covariance_matrix[0:new_num_pts,new_num_pts] = 1

                # Set up Right Hand Side (covariance between data and unknown)
                covariance_array = np.zeros(shape=(new_num_pts+1)) 
                k_weights = np.zeros(shape=(new_num_pts+1))
                covariance_array[0:new_num_pts] = Covariance.make_covariance_array(xy_val, 
                                                                        np.tile(prediction_grid[z], new_num_pts), 
                                                                        vario, rotation_matrix)
                covariance_array[new_num_pts] = 1 # unbiasedness constraint
                covariance_matrix.reshape(((new_num_pts+1)), ((new_num_pts+1)))
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array, rcond = None) # Calculate Kriging Weights   
                
                # to avoid underestimating variance, calculate weights for variance estimation separately
                vario2 = [vario[0], vario[1], vario[2], vario[3], VAR_1]
                covariance_array2 = np.zeros(shape=(new_num_pts+1))
                covariance_array2[new_num_pts] = 1 # unbiasedness constraint
                covariance_array2[0:new_num_pts] = Covariance.make_covariance_array(xy_val, 
                                                         np.tile(prediction_grid[z], new_num_pts), 
                                                         vario2, rotation_matrix)
                k_weights2, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array2, rcond = None)
                
                est_ok[z] = local_mean + np.sum(k_weights[0:new_num_pts]*(norm_data_val[:] - local_mean)) # get estimates
                var_ok[z] = VAR_1 - np.sum(k_weights2[0:new_num_pts]*covariance_array2[0:new_num_pts])
            else:
                est_ok[z] = df['Z'].values[np.where(test_idx==2)[0][0]]
                var_ok[z] = 0

        return est_ok, var_ok


    # sequential Gaussian simulation using ordinary kriging
    def skrige_sgs(prediction_grid, df, xx, yy, zz, NUM_POINTS, vario, RADIUS):

        """Sequential Gaussian simulation
        :param prediction_grid: x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
        :df: data frame of input data
        :xx: column name for x coordinates of input data frame
        :yy: column name for y coordinates of input data frame
        :df: column name for the data variable (in this case, bed elevation) of the input data frame
        :NUM_POINTS: the number of conditioning points to search for
        :vario: variogram parameters describing the spatial statistics
        """

        # unpack variogram parameters
        AZIMUTH = vario[0]
        MAJOR_RANGE = vario[2]
        MINOR_RANGE = vario[3]
        rotation_matrix = make_rotation_matrix(AZIMUTH, MAJOR_RANGE, MINOR_RANGE) # rotation matrix for scaling distance based on ranges and anisotropy

        # rename header names for consistency with other functions
        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})
        xyindex = np.arange(len(prediction_grid)) # generate random array for simulation order
        random.shuffle(xyindex)
        MEAN_1 = np.average(df["Z"].values) # mean of input data
        VAR_1 = np.var(df["Z"].values) # variance of data   
        sgs = np.zeros(shape=len(prediction_grid))  # preallocate space for simulation

        for idx, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            z = xyindex[idx] # get coordinate index
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0: # not our hard data
                
                # get nearest neighbors
                nearest = NearestNeighbor.nearest_neighbor_search(RADIUS, NUM_POINTS, 
                                                  prediction_grid[z], df[['X','Y','Z']])  
                norm_data_val = nearest[:,-1]   # store data values in new array
                xy_val = nearest[:,:-1]   # store X,Y pair values in new array
                new_num_pts = len(nearest) 

                # left hand side (covariance between data)
                covariance_matrix = np.zeros(shape=((new_num_pts, new_num_pts))) # left hand side (covariance between data)
                covariance_matrix = Covariance.make_covariance_matrix(xy_val, vario, rotation_matrix)

                # Set up Right Hand Side (covariance between data and unknown)
                covariance_array = np.zeros(shape=(new_num_pts)) 
                k_weights = np.zeros(shape=(new_num_pts))
                covariance_array = Covariance.make_covariance_array(xy_val, 
                                                         np.tile(prediction_grid[z], new_num_pts), 
                                                         vario, rotation_matrix)
                covariance_matrix.reshape(((new_num_pts)), ((new_num_pts)))
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array, rcond = None) # Calculate Kriging Weights
                # get estimates
                est = MEAN_1 + np.sum(k_weights*(norm_data_val - MEAN_1)) # simple kriging mean
                var = VAR_1 - np.sum(k_weights*covariance_array) # simple kriging variance
                var = np.absolute(var) # make sure variances are non-negative
                sgs[z] = np.random.normal(est,math.sqrt(var),1) # simulate by randomly sampling a value
            else:
                sgs[z] = df['Z'].values[np.where(test_idx==2)[0][0]]

            coords = prediction_grid[z:z+1,:] # update the conditioning data
            df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 
                                             'Z': [sgs[z]]})], sort=False) # add new points by concatenating dataframes 

        return sgs


    # sequential Gaussian simulation using ordinary kriging
    def okrige_sgs(prediction_grid, df, xx, yy, zz, NUM_POINTS, vario, RADIUS):

        """Sequential Gaussian simulation
        :param prediction_grid: x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
        :df: data frame of input data
        :xx: column name for x coordinates of input data frame
        :yy: column name for y coordinates of input data frame
        :df: column name for the data variable (in this case, bed elevation) of the input data frame
        :NUM_POINTS: the number of conditioning points to search for
        :vario: variogram parameters describing the spatial statistics
        """

        # unpack variogram parameters
        AZIMUTH = vario[0]
        MAJOR_RANGE = vario[2]
        MINOR_RANGE = vario[3]
        rotation_matrix = make_rotation_matrix(AZIMUTH, MAJOR_RANGE, MINOR_RANGE) # rotation matrix for scaling distance based on ranges and anisotropy

        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"}) # rename header names for consistency with other functions
        xyindex = np.arange(len(prediction_grid)) # generate random array for simulation order
        random.shuffle(xyindex)
        VAR_1 = np.var(df["Z"].values) # variance of data       
        sgs = np.zeros(shape=len(prediction_grid))  # preallocate space for simulation  

        for idx, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            z = xyindex[idx] # get coordinate index
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0: # not our hard data
                
                # gather nearest neighbor points
                nearest = NearestNeighbor.nearest_neighbor_search(RADIUS, NUM_POINTS, 
                                                  prediction_grid[z], df[['X','Y','Z']]) 
                norm_data_val = nearest[:,-1]   # store data values in new array
                xy_val = nearest[:,:-1]   # store X,Y pair values in new array
                local_mean = np.mean(norm_data_val) # compute the local mean
                new_num_pts = len(nearest) 

                covariance_matrix = np.zeros(shape=((new_num_pts+1, new_num_pts+1))) # left hand side (covariance between data)
                covariance_matrix[0:new_num_pts,0:new_num_pts] = Covariance.make_covariance_matrix(xy_val, 
                                                                                        vario, rotation_matrix)
                covariance_matrix[new_num_pts,0:new_num_pts] = 1
                covariance_matrix[0:new_num_pts,new_num_pts] = 1

                # Set up Right Hand Side (covariance between data and unknown)
                covariance_array = np.zeros(shape=(new_num_pts+1))
                k_weights = np.zeros(shape=(new_num_pts+1))
                covariance_array[0:new_num_pts] = Covariance.make_covariance_array(xy_val, 
                                                                        np.tile(prediction_grid[z], new_num_pts), 
                                                                        vario, rotation_matrix)
                covariance_array[new_num_pts] = 1 # unbiasedness constraint
                covariance_matrix.reshape(((new_num_pts+1)), ((new_num_pts+1)))

                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array, rcond = None) # Calculate Kriging Weights           
                est = local_mean + np.sum(k_weights[0:new_num_pts]*(norm_data_val - local_mean)) # kriging mean
                var = VAR_1 - np.sum(k_weights[0:new_num_pts]*covariance_array[0:new_num_pts]) # kriging variance
                var = np.absolute(var) # make sure variances are non-negative

                sgs[z] = np.random.normal(est,math.sqrt(var),1) # simulate by randomly sampling a value
            else:
                sgs[z] = df['Z'].values[np.where(test_idx==2)[0][0]] 

            # update the conditioning data
            coords = prediction_grid[z:z+1,:]
            df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 'Z': [sgs[z]]})], sort=False) # add new points by concatenating dataframes 

        return sgs

    # sgsim with multiple clusters
    def cluster_sgs(prediction_grid, df, xx, yy, zz, kk, NUM_POINTS, df_gamma, RADIUS):

        """Sequential Gaussian simulation
        :param prediction_grid: x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
        :df: data frame of input data
        :xx: column name for x coordinates of input data frame
        :yy: column name for y coordinates of input data frame
        :zz: y variable
        :kk: k-means cluster number
        :df: column name for the data variable (in this case, bed elevation) of the input data frame
        :NUM_POINTS: the number of conditioning points to search for
        :df_gamma: dataframe of variogram parameters describing the spatial statistics for each cluster
        """   

        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z", kk: "K"}) # rename header names for consistency with other functions 
        xyindex = np.arange(len(prediction_grid)) # generate random array for simulation order
        random.shuffle(xyindex)
        MEAN_1 = np.average(df["Z"].values) # mean of input data
        VAR_1 = np.var(df["Z"].values); # variance of data       
        sgs = np.zeros(shape=len(prediction_grid))  # preallocate space for simulation    

        for idx, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            z = xyindex[idx] # get coordinate index
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0: # not our hard data
                
                # gather nearest neighbor points and K cluster value
                nearest, cluster_number = NearestNeighbor.nearest_neighbor_search_cluster(RADIUS, 
                                                                                          NUM_POINTS, 
                                                                                          prediction_grid[z],
                                                                                          df[['X','Y','Z','K']])  
                vario = df_gamma.Variogram[cluster_number] # define variogram parameters using cluster value
                norm_data_val = nearest[:,-1]   # store data values in new array
                xy_val = nearest[:,:-1]   # store X,Y pair values in new array

                # unpack variogram parameters
                AZIMUTH = vario[0]
                MAJOR_RANGE = vario[2]
                MINOR_RANGE = vario[3]
                rotation_matrix = make_rotation_matrix(AZIMUTH, MAJOR_RANGE, MINOR_RANGE) # rotation matrix for scaling distance based on ranges and anisotropy
                new_num_pts = len(nearest)

                covariance_matrix = np.zeros(shape=((new_num_pts, new_num_pts))) # left hand side (covariance between data)
                covariance_matrix[0:new_num_pts,0:new_num_pts] = Covariance.make_covariance_matrix(xy_val, 
                                                                                                   vario, 
                                                                                                   rotation_matrix) 

                covariance_array = np.zeros(shape=(new_num_pts)) # Set up Right Hand Side (covariance between data and unknown)
                k_weights = np.zeros(shape=(new_num_pts))
                covariance_array = Covariance.make_covariance_array(xy_val, np.tile(prediction_grid[z], new_num_pts), 
                                                                    vario, rotation_matrix)
                covariance_matrix.reshape(((new_num_pts)), ((new_num_pts)))
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond = None) 
                est = MEAN_1 + np.sum(k_weights*(norm_data_val - MEAN_1)) # simple kriging mean
                var = VAR_1 - np.sum(k_weights*covariance_array) # simple kriging variance
                var = np.absolute(var) # make sure variances are non-negative

                sgs[z] = np.random.normal(est,math.sqrt(var),1) # simulate by randomly sampling a value
            else:
                sgs[z] = df['Z'].values[np.where(test_idx==2)[0][0]] 

            coords = prediction_grid[z:z+1,:] # update the conditioning data
            df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 
                                             'Z': [sgs[z]], 'K': [cluster_number]})], sort=False) 

        return sgs


    ###########################

    # Multivariate

    ###########################

    # perform simple collocated cokriging with MM1
    def cokrige_mm1(prediction_grid, df1, xx1, yy1, zz1, df2, xx2, yy2, zz2, NUM_POINTS, vario, RADIUS, CORRCOEF):

        # unpack variogram parameters for rotation matrix
        AZIMUTH = vario[0]
        MAJOR_RANGE = vario[2]
        MINOR_RANGE = vario[3]
        rotation_matrix = make_rotation_matrix(AZIMUTH, MAJOR_RANGE, MINOR_RANGE) # rotation matrix for scaling distance based on ranges and anisotropy

        df1 = df1.rename(columns = {xx1: "X", yy1: "Y", zz1: "Z"}) # rename header names for consistency with other functions
        df2 = df2.rename(columns = {xx2: "X", yy2: "Y", zz2: "Z"})

        MEAN_1 = np.average(df1['Z']) # mean of primary data
        VAR_1 = np.var(df1['Z']); # variance of primary data 
        MEAN_2 = np.average(df2['Z']) # mean of secondary data
        VAR_2 = np.var(df2['Z']); # variance of secondary data 

        est_cokrige = np.zeros(shape=len(prediction_grid)) # make zeros array the size of the prediction grid
        var_cokrige = np.zeros(shape=len(prediction_grid))

        for z, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            test_idx = np.sum(prediction_grid[z]==df1[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0: # not our hard data 
                
                # get nearest neighbors
                nearest = NearestNeighbor.nearest_neighbor_search(RADIUS, NUM_POINTS, 
                                                                  prediction_grid[z], 
                                                                  df1[['X','Y','Z']])           
                nearest_second = NearestNeighbor.nearest_neighbor_secondary(prediction_grid[z], 
                                                                            df2[['X','Y','Z']]) 
                norm_data_val = nearest[:,-1] # values of nearest neighbor points
                norm_data_val = norm_data_val.reshape(len(norm_data_val),1)
                norm_data_val = norm_data_val.T
                norm_data_val = np.append(norm_data_val, [nearest_second[-1]]) # append secondary data value
                xy_val = nearest[:, :-1] # coordinates of nearest neighbor points
                xy_second = nearest_second[:-1] # secondary data coordinates
                xy_val = np.append(xy_val, [xy_second], axis = 0) # append coordinates of secondary data
                new_num_pts = len(nearest)

                # covariance between data points
                covariance_matrix = np.zeros(shape=((new_num_pts + 1, new_num_pts + 1)))
                covariance_matrix[0:new_num_pts+1, 0:new_num_pts+1] = Covariance.make_covariance_matrix(xy_val, vario, rotation_matrix) # covariance within primary data

                # covariance between data and unknown
                covariance_array = np.zeros(shape=(new_num_pts + 1)) 
                k_weights = np.zeros(shape=(new_num_pts + 1))
                covariance_array[0:new_num_pts+1] = Covariance.make_covariance_array(xy_val, 
                                                                                     np.tile(prediction_grid[z], 
                                                                                             new_num_pts + 1), 
                                                                                     vario, rotation_matrix)
                covariance_array[new_num_pts] = covariance_array[new_num_pts] * CORRCOEF # correlation between primary and nearest neighbor (zero lag) secondary data

                # update covariance matrix with secondary info (gamma2 = rho12 * gamma1)
                covariance_matrix[new_num_pts, 0 : new_num_pts+1] = covariance_matrix[new_num_pts, 0 : new_num_pts+1] * CORRCOEF
                covariance_matrix[0 : new_num_pts+1, new_num_pts] = covariance_matrix[0 : new_num_pts+1, new_num_pts] * CORRCOEF
                covariance_matrix[new_num_pts, new_num_pts] = 1
                covariance_matrix.reshape(((new_num_pts + 1)), ((new_num_pts + 1)))

                # solve kriging system
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond = None) # get weights
                part1 = MEAN_1 + np.sum(k_weights[0:new_num_pts]*(norm_data_val[0:new_num_pts] - MEAN_1)/np.sqrt(VAR_1))
                part2 = k_weights[new_num_pts] * (nearest_second[-1] - MEAN_2)/np.sqrt(VAR_2)
                               
                est_cokrige[z] = part1 + part2 # compute mean
                var_cokrige[z] = 1 - np.sum(k_weights[0:new_num_pts+1]*covariance_array[0:new_num_pts+1]) # compute variance
            else:
                est_cokrige[z] = df1['Z'].values[np.where(test_idx==2)[0][0]]
                var_cokrige[z] = 0

        return est_cokrige, var_cokrige


    # perform cosimulation with MM1
    def cosim_mm1(prediction_grid, df1, xx1, yy1, zz1, df2, xx2, yy2, zz2, NUM_POINTS, vario, RADIUS, CORRCOEF):

        # unpack variogram parameters
        AZIMUTH = vario[0]
        MAJOR_RANGE = vario[2]
        MINOR_RANGE = vario[3]
        rotation_matrix = make_rotation_matrix(AZIMUTH, MAJOR_RANGE, MINOR_RANGE) # rotation matrix for scaling distance based on ranges and anisotropy

        df1 = df1.rename(columns = {xx1: "X", yy1: "Y", zz1: "Z"}) # rename header names for consistency with other functions
        df2 = df2.rename(columns = {xx2: "X", yy2: "Y", zz2: "Z"})
        xyindex = np.arange(len(prediction_grid)) # generate random array for simulation order
        random.shuffle(xyindex)

        MEAN_1 = np.average(df1['Z']) # mean of primary data
        VAR_1 = np.var(df1['Z']); # variance of primary data 
        MEAN_2 = np.average(df2['Z']) # mean of secondary data
        VAR_2 = np.var(df2['Z']); # variance of secondary data 

        #est_cokrige = np.zeros(shape=len(prediction_grid)) # make zeros array the size of the prediction grid
        #var_cokrige = np.zeros(shape=len(prediction_grid))    
        cosim = np.zeros(shape=len(prediction_grid))  # preallocate space for simulation

        # for each coordinate in the prediction grid
        for idx, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            z = xyindex[idx] # get coordinate index
            test_idx = np.sum(prediction_grid[z]==df1[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0: # not our hard data  
                
                # get nearest neighbors
                nearest = NearestNeighbor.nearest_neighbor_search(RADIUS, NUM_POINTS, 
                                                                  prediction_grid[z], 
                                                                  df1[['X','Y','Z']]) 
                nearest_second = NearestNeighbor.nearest_neighbor_secondary(prediction_grid[z], 
                                                                            df2[['X','Y','Z']])
                norm_data_val = nearest[:,-1] # values of nearest neighbor points
                norm_data_val = norm_data_val.reshape(len(norm_data_val),1)
                norm_data_val = norm_data_val.T
                norm_data_val = np.append(norm_data_val, [nearest_second[-1]]) # append secondary data value
                xy_val = nearest[:, :-1] # coordinates of nearest neighbor points
                xy_second = nearest_second[:-1] # secondary data coordinates
                xy_val = np.append(xy_val, [xy_second], axis = 0) # append coordinates of secondary data
                new_num_pts = len(nearest)

                # covariance between data poitns
                covariance_matrix = np.zeros(shape=((new_num_pts + 1, new_num_pts + 1))) # set up covariance matrix
                covariance_matrix[0:new_num_pts+1, 0:new_num_pts+1] = Covariance.make_covariance_matrix(xy_val, 
                                                                                                        vario, rotation_matrix) 

                # covariance between data and unknown
                covariance_array = np.zeros(shape=(new_num_pts + 1)) # get covariance between data and unknown grid cell
                k_weights = np.zeros(shape=(new_num_pts + 1))
                covariance_array[0:new_num_pts+1] = Covariance.make_covariance_array(xy_val, 
                                                                                     np.tile(prediction_grid[z], 
                                                                                             new_num_pts + 1), 
                                                                                     vario, rotation_matrix)
                covariance_array[new_num_pts] = covariance_array[new_num_pts] * CORRCOEF 

                # update covariance matrix with secondary info (gamma2 = rho12 * gamma1)
                covariance_matrix[new_num_pts, 0 : new_num_pts+1] = covariance_matrix[new_num_pts, 0 : new_num_pts+1] * CORRCOEF
                covariance_matrix[0 : new_num_pts+1, new_num_pts] = covariance_matrix[0 : new_num_pts+1, new_num_pts] * CORRCOEF
                covariance_matrix[new_num_pts, new_num_pts] = 1
                covariance_matrix.reshape(((new_num_pts + 1)), ((new_num_pts + 1)))

                # solve kriging system
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond = None) # get weights
                part1 = MEAN_1 + np.sum(k_weights[0:new_num_pts]*(norm_data_val[0:new_num_pts] - MEAN_1)/np.sqrt(VAR_1))
                part2 = k_weights[new_num_pts] * (nearest_second[-1] - MEAN_2)/np.sqrt(VAR_2)
                est_cokrige = part1 + part2 # compute mean
                var_cokrige = 1 - np.sum(k_weights*covariance_array) # compute variance
                var_cokrige = np.absolute(var_cokrige) # make sure variances are non-negative

                cosim[z] = np.random.normal(est_cokrige,math.sqrt(var_cokrige),1) # simulate by randomly sampling a value
            else:
                cosim[z] = df1['Z'].values[np.where(test_idx==2)[0][0]] 

            coords = prediction_grid[z:z+1,:] # update the conditioning data
            df1 = pd.concat([df1,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 'Z': [cosim[z]]})], sort=False) # add new points by concatenating dataframes 

        return cosim

# In[ ]:





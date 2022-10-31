#!/usr/bin/env python
# coding: utf-8

### geostatistical tools

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

class Gridding:

    def prediction_grid(xmin, xmax, ymin, ymax, res):
        """
        Make prediction grid
        Inputs:
            xmin - minimum x extent
            xmax - maximum x extent
            ymin - minimum y extent
            ymax - maximum y extent
            res - grid cell resolution
        Outputs:
            prediction_grid_xy - x,y array of coordinates
        """ 
        cols = np.rint((xmax - xmin + res)/res)
        rows = np.rint((ymax - ymin + res)/res)  
        x = np.linspace(xmin, xmin+(cols*res), num=int(cols), endpoint=False)
        y = np.linspace(ymin, ymin+(rows*res), num=int(rows), endpoint=False)
        xx, yy = np.meshgrid(x,y) 
        yy = np.flip(yy) 
        x = np.reshape(xx, (int(rows)*int(cols), 1))
        y = np.reshape(yy, (int(rows)*int(cols), 1))
        prediction_grid_xy = np.concatenate((x,y), axis = 1)
        
        return prediction_grid_xy
 
    def make_grid(xmin, xmax, ymin, ymax, res):
        """
        Generate coordinates for output of gridded data  
        Inputs:
            xmin - minimum x extent
            xmax - maximum x extent
            ymin - minimum y extent
            ymax - maximum y extent
            res - grid cell resolution
        Outputs:
            prediction_grid_xy - x,y array of coordinates
            rows - number of rows 
            cols - number of columns
        """ 
        cols = np.rint((xmax - xmin)/res) 
        rows = np.rint((ymax - ymin)/res)  
        rows = rows.astype(int)
        cols = cols.astype(int)
        x = np.arange(xmin,xmax,res); y = np.arange(ymin,ymax,res)
        xx, yy = np.meshgrid(x,y) 
        x = np.reshape(xx, (int(rows)*int(cols), 1)) 
        y = np.reshape(yy, (int(rows)*int(cols), 1))
        prediction_grid_xy = np.concatenate((x,y), axis = 1)
        
        return prediction_grid_xy, cols, rows
    
    def grid_data(df, xx, yy, zz, res):
        """
        Grid conditioning data
        Inputs:
            df - dataframe of conditioning data
            xx - column name for x coordinates of input data frame
            yy - column name for y coordinates of input data frame
            zz - column for z values (or data variable) of input data frame
            res - grid cell resolution
        Outputs:
            df_grid - dataframe of gridded data
            grid_matrix - matrix of gridded data
            rows - number of rows in grid_matrix
            cols - number of columns in grid_matrix
        """ 
        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})

        xmin = df['X'].min()
        xmax = df['X'].max()
        ymin = df['Y'].min()
        ymax = df['Y'].max()
        
        # make array of grid coordinates
        grid_coord, cols, rows = Gridding.make_grid(xmin, xmax, ymin, ymax, res) 

        df = df[['X','Y','Z']] 
        np_data = df.to_numpy() 
        np_resize = np.copy(np_data) 
        origin = np.array([xmin,ymin])
        resolution = np.array([res,res])
        
        # shift and re-scale the data by subtracting origin and dividing by resolution
        np_resize[:,:2] = np.rint((np_resize[:,:2]-origin)/resolution) 

        grid_sum = np.zeros((rows,cols))
        grid_count = np.copy(grid_sum) 

        for i in range(np_data.shape[0]):
            xindex = np.int32(np_resize[i,1])
            yindex = np.int32(np_resize[i,0])

            if ((xindex >= rows) | (yindex >= cols)):
                continue

            grid_sum[xindex,yindex] = np_data[i,2] + grid_sum[xindex,yindex]
            grid_count[xindex,yindex] = 1 + grid_count[xindex,yindex]


        np.seterr(invalid='ignore') 
        grid_matrix = np.divide(grid_sum, grid_count) 
        grid_array = np.reshape(grid_matrix,[rows*cols]) 
        grid_sum = np.reshape(grid_sum,[rows*cols]) 
        grid_count = np.reshape(grid_count,[rows*cols]) 

        # make dataframe    
        grid_total = np.array([grid_coord[:,0], grid_coord[:,1], 
                               grid_sum, grid_count, grid_array])    
        df_grid = pd.DataFrame(grid_total.T, 
                               columns = ['X', 'Y', 'Sum', 'Count', 'Z']) 
        grid_matrix = np.flipud(grid_matrix) 
        
        return df_grid, grid_matrix, rows, cols


###################################

# RBF trend estimation

###################################

def rbf_trend(grid_matrix, smooth_factor, res):
    """
    Estimate trend using radial basis functions
    Inputs:
        grid_matrix - matrix of gridded conditioning data
        smooth_factor - regularizing parameter
        res - grid cell resolution
    Outputs:
        trend_rbf - trend estimate
    """ 
    sigma = np.rint(smooth_factor/res)
    ny, nx = grid_matrix.shape
    rbfi = Rbf(np.where(~np.isnan(grid_matrix))[1],
               np.where(~np.isnan(grid_matrix))[0], 
               grid_matrix[~np.isnan(grid_matrix)],smooth = sigma)

    # evaluate RBF
    yi = np.arange(nx)
    xi = np.arange(ny)
    xi,yi = np.meshgrid(xi, yi)
    trend_rbf = rbfi(xi, yi)   
    
    return trend_rbf


####################################

# Nearest neighbor octant search

####################################

class NearestNeighbor:

    def center(arrayx, arrayy, centerx, centery):
        """
        Shift data points so that grid cell of interest is at the origin
        Inputs:
            arrayx - x coordinates of data
            arrayy - y coordinates of data
            centerx - x coordinate of grid cell of interest
            centery - y coordinate of grid cell of interest
        Outputs:
            centered_array - array of coordinates that are shifted with respect to grid cell of interest
        """ 
        centerx = arrayx - centerx
        centery = arrayy - centery
        centered_array = np.array([centerx, centery])
        
        return centered_array

    def distance_calculator(centered_array):
        """
        Compute distances between coordinates and the origin
        Inputs:
            centered_array - array of coordinates
        Outputs:
            dist - array of distances between coordinates and origin
        """ 
        dist = np.linalg.norm(centered_array, axis=0)
        
        return dist

    def angle_calculator(centered_array):
        """
        Compute angles between coordinates and the origin
        Inputs:
            centered_array - array of coordinates
        Outputs:
            angles - array of angles between coordinates and origin
        """ 
        angles = np.arctan2(centered_array[0], centered_array[1])
        
        return angles
    
    def nearest_neighbor_search(radius, num_points, loc, data2):
        """
        Nearest neighbor octant search
        Inputs:
            radius - search radius
            num_points - number of points to search for
            loc - coordinates for grid cell of interest
            data2 - data 
        Outputs:
            near - nearest neighbors
        """ 
        locx = loc[0]
        locy = loc[1]
        data = data2.copy()
        centered_array = NearestNeighbor.center(data['X'].values, data['Y'].values, 
                                locx, locy)
        data["dist"] = NearestNeighbor.distance_calculator(centered_array) 
        data["angles"] = NearestNeighbor.angle_calculator(centered_array)
        data = data[data.dist < radius] 
        data = data.sort_values('dist', ascending = True)
        bins = [-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0, 
                math.pi/4, math.pi/2, 3*math.pi/4, math.pi] 
        data["Oct"] = pd.cut(data.angles, bins = bins, labels = list(range(8)))
        oct_count = num_points // 8
        smallest = np.ones(shape=(num_points, 3)) * np.nan

        for i in range(8):
            octant = data[data.Oct == i].iloc[:oct_count][['X','Y','Z']].values 
            for j, row in enumerate(octant):
                smallest[i*oct_count+j,:] = row 
        near = smallest[~np.isnan(smallest)].reshape(-1,3) 
        
        return near
    
    def nearest_neighbor_search_cluster(radius, num_points, loc, data2):
        """
        Nearest neighbor octant search when doing sgs with clusters
        Inputs:
            radius - search radius
            num_points - number of points to search for
            loc - coordinates for grid cell of interest
            data2 - data 
        Outputs:
            near - nearest neighbors
            cluster_number - nearest neighbor cluster number
        """ 
        locx = loc[0]
        locy = loc[1]
        data = data2.copy()
        centered_array = NearestNeighbor.center(data['X'].values, data['Y'].values, 
                                locx, locy)
        data["dist"] = NearestNeighbor.distance_calculator(centered_array)
        data["angles"] = NearestNeighbor.angle_calculator(centered_array)
        data = data[data.dist < radius] 
        data = data.sort_values('dist', ascending = True)
        data = data.reset_index() 
        cluster_number = data.K[0] 
        bins = [-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0, 
                math.pi/4, math.pi/2, 3*math.pi/4, math.pi]
        data["Oct"] = pd.cut(data.angles, bins = bins, labels = list(range(8))) 
        oct_count = num_points // 8
        smallest = np.ones(shape=(num_points, 3)) * np.nan

        for i in range(8):
            octant = data[data.Oct == i].iloc[:oct_count][['X','Y','Z']].values
            for j, row in enumerate(octant):
                smallest[i*oct_count+j,:] = row 
        near = smallest[~np.isnan(smallest)].reshape(-1,3) 
        
        return near, cluster_number

    def nearest_neighbor_secondary(loc, data2):
        """
        Find the neareset neighbor secondary data point to grid cell of interest
        Inputs:
            loc - coordinates for grid cell of interest
            data2 - secondary data
        Outputs:
            nearest_second - nearest neighbor value to secondary data
        """ 
        locx = loc[0]
        locy = loc[1]
        data = data2.copy()
        centered_array = NearestNeighbor.center(data['X'].values, data['Y'].values, 
                                locx, locy)
        data["dist"] = NearestNeighbor.distance_calculator(centered_array)
        data = data.sort_values('dist', ascending = True) 
        data = data.reset_index() 
        nearest_second = data.iloc[0][['X','Y','Z']].values 
        
        return nearest_second

    def find_colocated(df1, xx1, yy1, zz1, df2, xx2, yy2, zz2): 
        """
        Find colocated data between primary and secondary variables
        Inputs:
            df1 - data frame of primary conditioning data
            xx1 - column name for x coordinates of input data frame for primary data
            yy1 - column name for y coordinates of input data frame for primary data
            zz1 - column for z values (or data variable) of input data frame for primary data
            df2 - data frame of secondary data
            xx2 - column name for x coordinates of input data frame for secondary data
            yy2 - column name for y coordinates of input data frame for secondary data
            zz2 - column for z values (or data variable) of input data frame for secondary data
        Outputs:
            df_colocated - data frame of colocated values
        """ 
        df1 = df1.rename(columns = {xx1: "X", yy1: "Y", zz1: "Z"}) 
        df2 = df2.rename(columns = {xx2: "X", yy2: "Y", zz2: "Z"}) 
        secondary_variable_xy = df2[['X','Y']].values
        secondary_variable_tree = KDTree(secondary_variable_xy) 
        primary_variable_xy = df1[['X','Y']].values
        nearest_indices = np.zeros(len(primary_variable_xy)) 

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

###############################

# adaptive partioning

###############################

def adaptive_partitioning(df_data, xmin, xmax, ymin, ymax, i, max_points, min_length, max_iter=None):
    """
    Rercursively split clusters until they are all below max_points, but don't go smaller than min_length
    Inputs:
        df_data - DataFrame with X, Y, and K (cluster id)
        xmin - min x value of this partion
        xmax - max x value of this partion
        ymin - min y value of this partion
        ymax - max y value of this partion
        i - keeps track of total calls to this function
        max_points - all clusters will be "quartered" until points below this
        min_length - minimum side length of sqaures, preference over max_points
        max_iter - maximum iterations if worried about unending recursion
    Outputs:
        df_data - updated DataFrame with new cluster assigned the next integer
        i - number of iterations
    """
    # optional 'safety' if there is concern about runaway recursion
    if max_iter is not None:
        if i >= max_iter:
            return df_data, i
    
    dx = xmax - xmin
    dy = ymax - ymin
    
    # >= and <= greedy so we don't miss any points
    xleft = (df_data.X >= xmin) & (df_data.X <= xmin+dx/2)
    xright = (df_data.X <= xmax) & (df_data.X >= xmin+dx/2)
    ybottom = (df_data.Y >= ymin) & (df_data.Y <= ymin+dy/2)
    ytop = (df_data.Y <= ymax) & (df_data.Y >= ymin+dy/2)
    
    # index the current cell into 4 quarters
    q1 = df_data.loc[xleft & ybottom]
    q2 = df_data.loc[xleft & ytop]
    q3 = df_data.loc[xright & ytop]
    q4 = df_data.loc[xright & ybottom]
    
    # for each quarter, qaurter if too many points, else assign K and return
    for q in [q1, q2, q3, q4]:
        if (q.shape[0] > max_points) & (dx/2 > min_length):
            i = i+1
            df_data, i = adaptive_partitioning(df_data, q.X.min(), 
                                               q.X.max(), q.Y.min(), q.Y.max(), i, 
                                               max_points, min_length, max_iter)
        else:
            qcount = df_data.K.max()
            # ensure zero indexing
            if np.isnan(qcount) == True:
                qcount = 0
            else:
                qcount += 1
            df_data.loc[q.index, 'K'] = qcount
            
    return df_data, i


#########################

# Rotation Matrix

#########################

def make_rotation_matrix(azimuth, major_range, minor_range):
    """
    Make rotation matrix for accommodating anisotropy
    Inputs:
        azimuth - angle (in degrees from horizontal) of axis of orientation
        major_range - range parameter of variogram in major direction, or azimuth
        minor_range - range parameter of variogram in minor direction, or orthogonal to azimuth
    Outputs:
        rotation_matrix - 2x2 rotation matrix used to perform coordinate transformations
    """
    theta = (azimuth / 180.0) * np.pi 
    
    rotation_matrix = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],]),
        np.array([[1 / major_range, 0], [0, 1 / minor_range]]))
    
    return rotation_matrix


###########################

# Covariance functions

###########################

class Covariance:

    def covar(effective_lag, sill, nug):
        """
        Compute covariance using exponential covariance model
        Inputs:
            effective_lag - lag distance that is normalized to a range of 1
            sill - sill of variogram
            nug - nugget of variogram
        Outputs:
            c - covariance
        """
        c = (sill - nug)*np.exp(-3 * effective_lag)

        return c

    def make_covariance_matrix(coord, vario, rotation_matrix):
        """
        Make covariance matrix showing covariances between each pair of input coordinates
        Inputs:
            coord - coordinates of data points
            vario - array of variogram parameters
            rotation_matrix - rotation matrix used to perform coordinate transformations
        Outputs:
            covariance_matrix - nxn matrix of covariance between n points
        """
        nug = vario[1]
        sill = vario[4]   
        mat = np.matmul(coord, rotation_matrix)
        effective_lag = pairwise_distances(mat,mat) 
        covariance_matrix = Covariance.covar(effective_lag, sill, nug) 

        return covariance_matrix

    def make_covariance_array(coord1, coord2, vario, rotation_matrix):
        """
        Make covariance array showing covariances between each data points and grid cell of interest
        Inputs:
            coord1 - coordinates of n data points
            coord2 - coordinates of grid cell of interest (i.e. grid cell being simulated) that is repeated n times
            vario - array of variogram parameters
            rotation_matrix - rotation matrix used to perform coordinate transformations
        Outputs:
            covariance_array - nx1 array of covariance between n points and grid cell of interest
        """
        nug = vario[1]
        sill = vario[4]
        mat1 = np.matmul(coord1, rotation_matrix) 
        mat2 = np.matmul(coord2.reshape(-1,2), rotation_matrix) 
        effective_lag = np.sqrt(np.square(mat1 - mat2).sum(axis=1))
        covariance_array = Covariance.covar(effective_lag, sill, nug)

        return covariance_array

######################################

# Simple Kriging Function

######################################

class Interpolation: 

    def skrige(prediction_grid, df, xx, yy, zz, num_points, vario, radius):
        """
        Simple kriging interpolation
        Inputs:
            prediction_grid - x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df - data frame of conditioning data
            xx - column name for x coordinates of input data frame
            yy - column name for y coordinates of input data frame
            zz - column for z values (or data variable) of input data frame
            num_points - the number of conditioning points to search for
            vario - numpy array of variogram parameters describing the spatial statistics
            radius - search radius
        Outputs:
            est_sk - simple kriging estimate for each coordinate in prediction_grid
            var_sk - simple kriging variance 
        """
        
        # unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range) 

        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})   
        mean_1 = df['Z'].mean() 
        var_1 = vario[4]
        est_sk = np.zeros(shape=len(prediction_grid)) 
        var_sk = np.zeros(shape=len(prediction_grid))

        # for each coordinate in the prediction grid
        for z, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0:
                
                # gather nearest points within radius
                nearest = NearestNeighbor.nearest_neighbor_search(radius, num_points, 
                                                  prediction_grid[z], df[['X','Y','Z']])
                norm_data_val = nearest[:,-1] 
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

                est_sk[z] = mean_1 + (np.sum(k_weights*(norm_data_val[:] - mean_1))) 
                var_sk[z] = var_1 - np.sum(k_weights*covariance_array)
            else:
                est_sk[z] = df['Z'].values[np.where(test_idx==2)[0][0]]
                var_sk[z] = 0

        return est_sk, var_sk

    def okrige(prediction_grid, df, xx, yy, zz, num_points, vario, radius):
        """
        Ordinary kriging interpolation
        Inputs:
            prediction_grid - x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df - data frame of conditioning data
            xx - column name for x coordinates of input data frame
            yy - column name for y coordinates of input data frame
            zz - column for z values (or data variable) of input data frame
            num_points - the number of conditioning points to search for
            vario - numpy array of variogram parameters describing the spatial statistics
            radius - search radius
        Outputs:
            est_ok - ordinary kriging estimate for each coordinate in prediction_grid
            var_ok - ordinary kriging variance 
        """

        # unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range) 

        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"}) 
        var_1 = vario[4]
        est_ok = np.zeros(shape=len(prediction_grid)) 
        var_ok = np.zeros(shape=len(prediction_grid))

        for z, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0: 
                
                # find nearest data points
                nearest = NearestNeighbor.nearest_neighbor_search(radius, num_points, 
                                                  prediction_grid[z], df[['X','Y','Z']])    
                norm_data_val = nearest[:,-1] 
                local_mean = np.mean(norm_data_val) 
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
                covariance_array[new_num_pts] = 1 
                covariance_matrix.reshape(((new_num_pts+1)), ((new_num_pts+1)))
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array, rcond = None) 
                
                est_ok[z] = local_mean + np.sum(k_weights[0:new_num_pts]*(norm_data_val[:] - local_mean)) 
                var_ok[z] = var_1 - np.sum(k_weights[0:new_num_pts]*covariance_array[0:new_num_pts])
            else:
                est_ok[z] = df['Z'].values[np.where(test_idx==2)[0][0]]
                var_ok[z] = 0

        return est_ok, var_ok
  
    def skrige_sgs(prediction_grid, df, xx, yy, zz, num_points, vario, radius):
        """
        Sequential Gaussian simulation using simple kriging 
        Inputs:
            prediction_grid - x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df - data frame of conditioning data
            xx - column name for x coordinates of input data frame
            yy - column name for y coordinates of input data frame
            zz - column for z values (or data variable) of input data frame
            num_points - the number of conditioning points to search for
            vario - numpy array of variogram parameters describing the spatial statistics
            radius - search radius
        Outputs:
            sgs - simulated value for each coordinate in prediction_grid
        """

        # unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range) 

        df = df.rename(columns = {xx: 'X', yy: 'Y', zz: 'Z'})
        xyindex = np.arange(len(prediction_grid)) 
        random.shuffle(xyindex)
        mean_1 = df['Z'].mean() 
        var_1 = vario[4]
        sgs = np.zeros(shape=len(prediction_grid)) 

        for idx, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            z = xyindex[idx] 
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values, axis=1)
            if np.sum(test_idx==2)==0: 
                
                # get nearest neighbors
                nearest = NearestNeighbor.nearest_neighbor_search(radius, num_points, 
                                                  prediction_grid[z], df[['X','Y','Z']])  
                norm_data_val = nearest[:,-1]   
                xy_val = nearest[:,:-1]   
                new_num_pts = len(nearest) 

                # left hand side (covariance between data)
                covariance_matrix = np.zeros(shape=((new_num_pts, new_num_pts))) 
                covariance_matrix = Covariance.make_covariance_matrix(xy_val, vario, rotation_matrix)

                # Set up Right Hand Side (covariance between data and unknown)
                covariance_array = np.zeros(shape=(new_num_pts)) 
                k_weights = np.zeros(shape=(new_num_pts))
                covariance_array = Covariance.make_covariance_array(xy_val, 
                                                         np.tile(prediction_grid[z], new_num_pts), 
                                                         vario, rotation_matrix)
                covariance_matrix.reshape(((new_num_pts)), ((new_num_pts)))
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array, rcond = None) 
                # get estimates
                est = mean_1 + np.sum(k_weights*(norm_data_val - mean_1)) 
                var = var_1 - np.sum(k_weights*covariance_array) 
                var = np.absolute(var) 
                sgs[z] = np.random.normal(est,math.sqrt(var),1) 
            else:
                sgs[z] = df['Z'].values[np.where(test_idx==2)[0][0]]

            coords = prediction_grid[z:z+1,:] 
            df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 
                                             'Z': [sgs[z]]})], sort=False) 

        return sgs
   
    def okrige_sgs(prediction_grid, df, xx, yy, zz, num_points, vario, radius):
        """
        Sequential Gaussian simulation using ordinary kriging 
        Inputs:
            prediction_grid - x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df - data frame of conditioning data
            xx - column name for x coordinates of input data frame
            yy - column name for y coordinates of input data frame
            zz - column for z values (or data variable) of input data frame
            num_points - the number of conditioning points to search for
            vario - numpy array of variogram parameters describing the spatial statistics
            radius - search radius
        Outputs:
            sgs - simulated value for each coordinate in prediction_grid
        """

        # unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range) 

        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"}) 
        xyindex = np.arange(len(prediction_grid)) 
        random.shuffle(xyindex)
        var_1 = vario[4]
        sgs = np.zeros(shape=len(prediction_grid))  

        for idx, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            z = xyindex[idx] 
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0:
                
                # gather nearest neighbor points
                nearest = NearestNeighbor.nearest_neighbor_search(radius, num_points, 
                                                  prediction_grid[z], df[['X','Y','Z']]) 
                norm_data_val = nearest[:,-1]   
                xy_val = nearest[:,:-1]   
                local_mean = np.mean(norm_data_val) 
                new_num_pts = len(nearest) 

                # covariance between data
                covariance_matrix = np.zeros(shape=((new_num_pts+1, new_num_pts+1))) 
                covariance_matrix[0:new_num_pts,0:new_num_pts] = Covariance.make_covariance_matrix(xy_val, 
                                                                                        vario, rotation_matrix)
                covariance_matrix[new_num_pts,0:new_num_pts] = 1
                covariance_matrix[0:new_num_pts,new_num_pts] = 1

                # covariance between data and unknown
                covariance_array = np.zeros(shape=(new_num_pts+1))
                k_weights = np.zeros(shape=(new_num_pts+1))
                covariance_array[0:new_num_pts] = Covariance.make_covariance_array(xy_val, 
                                                                        np.tile(prediction_grid[z], new_num_pts), 
                                                                        vario, rotation_matrix)
                covariance_array[new_num_pts] = 1 
                covariance_matrix.reshape(((new_num_pts+1)), ((new_num_pts+1)))

                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array, rcond = None)           
                est = local_mean + np.sum(k_weights[0:new_num_pts]*(norm_data_val - local_mean)) 
                var = var_1 - np.sum(k_weights[0:new_num_pts]*covariance_array[0:new_num_pts]) 
                var = np.absolute(var)

                sgs[z] = np.random.normal(est,math.sqrt(var),1) 
            else:
                sgs[z] = df['Z'].values[np.where(test_idx==2)[0][0]] 

            coords = prediction_grid[z:z+1,:]
            df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 'Z': [sgs[z]]})], sort=False) 

        return sgs


    def cluster_sgs(prediction_grid, df, xx, yy, zz, kk, num_points, df_gamma, radius):
        """
        Sequential Gaussian simulation where variogram parameters are different for each k cluster. Uses simple kriging 
        Inputs:
            prediction_grid - x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df - data frame of conditioning data
            xx - column name for x coordinates of input data frame
            yy - column name for y coordinates of input data frame
            zz - column for z values (or data variable) of input data frame
            kk - column of k cluster numbers for each point
            num_points - the number of conditioning points to search for
            df_gamma - dataframe with variogram parameters for each cluster
            radius - search radius
        Outputs:
            sgs - simulated value for each coordinate in prediction_grid
        """ 

        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z", kk: "K"})  
        xyindex = np.arange(len(prediction_grid)) 
        random.shuffle(xyindex)
        mean_1 = np.average(df["Z"].values) 
        sgs = np.zeros(shape=len(prediction_grid)) 

        for idx, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            z = xyindex[idx] 
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0: 
                
                # gather nearest neighbor points and K cluster value
                nearest, cluster_number = NearestNeighbor.nearest_neighbor_search_cluster(radius, 
                                                                                          num_points, 
                                                                                          prediction_grid[z],
                                                                                          df[['X','Y','Z','K']])  
                vario = df_gamma.Variogram[cluster_number] 
                norm_data_val = nearest[:,-1]   
                xy_val = nearest[:,:-1]   

                # unpack variogram parameters
                azimuth = vario[0]
                major_range = vario[2]
                minor_range = vario[3]
                var_1 = vario[4]
                rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range) 
                new_num_pts = len(nearest)

                # covariance between data
                covariance_matrix = np.zeros(shape=((new_num_pts, new_num_pts))) 
                covariance_matrix[0:new_num_pts,0:new_num_pts] = Covariance.make_covariance_matrix(xy_val, 
                                                                                                   vario, 
                                                                                                   rotation_matrix) 
                
                # covariance between data and unknown
                covariance_array = np.zeros(shape=(new_num_pts)) 
                k_weights = np.zeros(shape=(new_num_pts))
                covariance_array = Covariance.make_covariance_array(xy_val, np.tile(prediction_grid[z], new_num_pts), 
                                                                    vario, rotation_matrix)
                covariance_matrix.reshape(((new_num_pts)), ((new_num_pts)))
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond = None) 
                est = mean_1 + np.sum(k_weights*(norm_data_val - mean_1)) 
                var = var_1 - np.sum(k_weights*covariance_array)
                var = np.absolute(var) 

                sgs[z] = np.random.normal(est,math.sqrt(var),1) 
            else:
                sgs[z] = df['Z'].values[np.where(test_idx==2)[0][0]]
                cluster_number = df['K'].values[np.where(test_idx==2)[0][0]]

            coords = prediction_grid[z:z+1,:] 
            df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 
                                             'Z': [sgs[z]], 'K': [cluster_number]})], sort=False)

        return sgs

    def cokrige_mm1(prediction_grid, df1, xx1, yy1, zz1, df2, xx2, yy2, zz2, num_points, vario, radius, corrcoef):
        """
        Simple collocated cokriging under Markov model 1 assumptions
        Inputs:
            prediction_grid - x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df1 - data frame of primary conditioning data
            xx1 - column name for x coordinates of input data frame for primary data
            yy1 - column name for y coordinates of input data frame for primary data
            zz1 - column for z values (or data variable) of input data frame for primary data
            df2 - data frame of secondary data
            xx2 - column name for x coordinates of input data frame for secondary data
            yy2 - column name for y coordinates of input data frame for secondary data
            zz2 - column for z values (or data variable) of input data frame for secondary data
            num_points - the number of conditioning points to search for
            vario - numpy array of variogram parameters describing the spatial statistics
            radius - search radius
            corrcoef - correlation coefficient between primary and secondary data
        Outputs:
            est_cokrige - cokriging estimate for each point in coordinate grid
            var_cokrige - variances
        """
        
        # unpack variogram parameters for rotation matrix
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range) 

        df1 = df1.rename(columns = {xx1: "X", yy1: "Y", zz1: "Z"})
        df2 = df2.rename(columns = {xx2: "X", yy2: "Y", zz2: "Z"})

        mean_1 = np.average(df1['Z']) 
        var_1 = vario[4]
        mean_2 = np.average(df2['Z']) 
        var_2 = np.var(df2['Z'])

        est_cokrige = np.zeros(shape=len(prediction_grid)) 
        var_cokrige = np.zeros(shape=len(prediction_grid))

        for z, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            test_idx = np.sum(prediction_grid[z]==df1[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0: #
                
                # get nearest neighbors
                nearest = NearestNeighbor.nearest_neighbor_search(radius, num_points, 
                                                                  prediction_grid[z], 
                                                                  df1[['X','Y','Z']])           
                nearest_second = NearestNeighbor.nearest_neighbor_secondary(prediction_grid[z], 
                                                                            df2[['X','Y','Z']]) 
                norm_data_val = nearest[:,-1] 
                norm_data_val = norm_data_val.reshape(len(norm_data_val),1)
                norm_data_val = norm_data_val.T
                norm_data_val = np.append(norm_data_val, [nearest_second[-1]]) 
                xy_val = nearest[:, :-1] 
                xy_second = nearest_second[:-1] 
                xy_val = np.append(xy_val, [xy_second], axis = 0) 
                new_num_pts = len(nearest)

                # covariance between data points
                covariance_matrix = np.zeros(shape=((new_num_pts + 1, new_num_pts + 1)))
                covariance_matrix[0:new_num_pts+1, 0:new_num_pts+1] = Covariance.make_covariance_matrix(xy_val, 
                                                                                                        vario, rotation_matrix) 

                # covariance between data and unknown
                covariance_array = np.zeros(shape=(new_num_pts + 1)) 
                k_weights = np.zeros(shape=(new_num_pts + 1))
                covariance_array[0:new_num_pts+1] = Covariance.make_covariance_array(xy_val, 
                                                                                     np.tile(prediction_grid[z], 
                                                                                             new_num_pts + 1), 
                                                                                     vario, rotation_matrix)
                covariance_array[new_num_pts] = covariance_array[new_num_pts] * corrcoef 

                # update covariance matrix with secondary info (gamma2 = rho12 * gamma1)
                covariance_matrix[new_num_pts, 0 : new_num_pts+1] = covariance_matrix[new_num_pts, 0 : new_num_pts+1] * corrcoef
                covariance_matrix[0 : new_num_pts+1, new_num_pts] = covariance_matrix[0 : new_num_pts+1, new_num_pts] * corrcoef
                covariance_matrix[new_num_pts, new_num_pts] = 1
                covariance_matrix.reshape(((new_num_pts + 1)), ((new_num_pts + 1)))

                # solve kriging system
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond = None) 
                part1 = mean_1 + np.sum(k_weights[0:new_num_pts]*(norm_data_val[0:new_num_pts] - mean_1)/np.sqrt(var_1))
                part2 = k_weights[new_num_pts] * (nearest_second[-1] - mean_2)/np.sqrt(var_2)
                               
                est_cokrige[z] = part1 + part2 
                var_cokrige[z] = var_1 - np.sum(k_weights[0:new_num_pts+1]*covariance_array[0:new_num_pts+1]) 
            else:
                est_cokrige[z] = df1['Z'].values[np.where(test_idx==2)[0][0]]
                var_cokrige[z] = 0

        return est_cokrige, var_cokrige

    def cosim_mm1(prediction_grid, df1, xx1, yy1, zz1, df2, xx2, yy2, zz2, num_points, vario, radius, corrcoef):
        """
        Cosimulation under Markov model 1 assumptions
        Inputs:
            prediction_grid - x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df1 - data frame of primary conditioning data
            xx1 - column name for x coordinates of input data frame for primary data
            yy1 - column name for y coordinates of input data frame for primary data
            zz1 - column for z values (or data variable) of input data frame for primary data
            df2 - data frame of secondary data
            xx2 - column name for x coordinates of input data frame for secondary data
            yy2 - column name for y coordinates of input data frame for secondary data
            zz2 - column for z values (or data variable) of input data frame for secondary data
            num_points - the number of conditioning points to search for
            vario - numpy array of variogram parameters describing the spatial statistics
            radius - search radius
            corrcoef - correlation coefficient between primary and secondary data
        Outputs:
            cosim - cosimulation for each point in coordinate grid
        """
            
        # unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range)
        df1 = df1.rename(columns = {xx1: "X", yy1: "Y", zz1: "Z"}) 
        df2 = df2.rename(columns = {xx2: "X", yy2: "Y", zz2: "Z"})
        xyindex = np.arange(len(prediction_grid)) 
        random.shuffle(xyindex)

        mean_1 = np.average(df1['Z']) 
        var_1 = vario[4]
        mean_2 = np.average(df2['Z']) 
        var_2 = np.var(df2['Z'])
   
        cosim = np.zeros(shape=len(prediction_grid))

        # for each coordinate in the prediction grid
        for idx, predxy in enumerate(tqdm(prediction_grid, position=0, leave=True)):
            z = xyindex[idx]
            test_idx = np.sum(prediction_grid[z]==df1[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0:
                
                # get nearest neighbors
                nearest = NearestNeighbor.nearest_neighbor_search(radius, num_points, 
                                                                  prediction_grid[z], 
                                                                  df1[['X','Y','Z']]) 
                nearest_second = NearestNeighbor.nearest_neighbor_secondary(prediction_grid[z], 
                                                                            df2[['X','Y','Z']])
                norm_data_val = nearest[:,-1] 
                norm_data_val = norm_data_val.reshape(len(norm_data_val),1)
                norm_data_val = norm_data_val.T
                norm_data_val = np.append(norm_data_val, [nearest_second[-1]]) 
                xy_val = nearest[:, :-1] 
                xy_second = nearest_second[:-1] #
                xy_val = np.append(xy_val, [xy_second], axis = 0) 
                new_num_pts = len(nearest)

                # covariance between data poitns
                covariance_matrix = np.zeros(shape=((new_num_pts + 1, new_num_pts + 1))) 
                covariance_matrix[0:new_num_pts+1, 0:new_num_pts+1] = Covariance.make_covariance_matrix(xy_val, 
                                                                                                        vario, rotation_matrix) 

                # covariance between data and unknown
                covariance_array = np.zeros(shape=(new_num_pts + 1)) 
                k_weights = np.zeros(shape=(new_num_pts + 1))
                covariance_array[0:new_num_pts+1] = Covariance.make_covariance_array(xy_val, 
                                                                                     np.tile(prediction_grid[z], 
                                                                                             new_num_pts + 1), 
                                                                                     vario, rotation_matrix)
                covariance_array[new_num_pts] = covariance_array[new_num_pts] * corrcoef 

                # update covariance matrix with secondary info (gamma2 = rho12 * gamma1)
                covariance_matrix[new_num_pts, 0 : new_num_pts+1] = covariance_matrix[new_num_pts, 0 : new_num_pts+1] * corrcoef
                covariance_matrix[0 : new_num_pts+1, new_num_pts] = covariance_matrix[0 : new_num_pts+1, new_num_pts] * corrcoef
                covariance_matrix[new_num_pts, new_num_pts] = 1
                covariance_matrix.reshape(((new_num_pts + 1)), ((new_num_pts + 1)))

                # solve kriging system
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond = None) 
                part1 = mean_1 + np.sum(k_weights[0:new_num_pts]*(norm_data_val[0:new_num_pts] - mean_1)/np.sqrt(var_1))
                part2 = k_weights[new_num_pts] * (nearest_second[-1] - mean_2)/np.sqrt(var_2)
                est_cokrige = part1 + part2 
                var_cokrige = var_1 - np.sum(k_weights*covariance_array)
                var_cokrige = np.absolute(var_cokrige) 

                cosim[z] = np.random.normal(est_cokrige,math.sqrt(var_cokrige),1) 
            else:
                cosim[z] = df1['Z'].values[np.where(test_idx==2)[0][0]]

            coords = prediction_grid[z:z+1,:]
            df1 = pd.concat([df1,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 'Z': [cosim[z]]})], sort=False) 

        return cosim

__all__ = ['Gridding', 'NearestNeighbor', 'Covariance', 'Interpolation', 'rbf_trend', 
    'adaptive_partitioning', 'make_rotation_matrix']

def __dir__():
    return __all__

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f'module {__name__} has no attribute {name}')
    return globals()[name]
import pytest

import os

import pandas as pd
import numpy as np
import random

import gstatsim as gs


def test_ordinary_kriging():
    """
    Test of ordinary kriging.
    The test is roughly based on demos/3_Simple_kriging_and_ordinary_kriging.ipynb

    """
    # read demo data
    data_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../demos/data/greenland_test_data.csv')
    df_bed = pd.read_csv(data_file_path)

    # grid data to 100 m resolution and remove coordinates with NaNs
    res = 1000
    df_grid, _, _, __import__ = gs.Gridding.grid_data(
        df_bed, 'X', 'Y', 'Bed', res)
    df_grid = df_grid[df_grid["Z"].isnull() == False]

    # define coordinate grid
    xmin = np.min(df_grid['X'])
    xmax = np.max(df_grid['X'])     # min and max x values
    ymin = np.min(df_grid['Y'])
    ymax = np.max(df_grid['Y'])     # min and max y values

    Pred_grid_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)

    # set random seed
    np.random.seed(42)

    # pick ("random") points from grid
    index_points = np.random.choice(
        range(len(Pred_grid_xy)), size=25, replace=False)
    Pred_grid_xy = Pred_grid_xy[index_points, :]

    # set variogram parameters
    azimuth = 0
    nugget = 0

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = 19236.
    minor_range = 19236.
    sill = 22399.
    vtype = 'Exponential'

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

    k = 100         # number of neighboring data points used to estimate a given point
    rad = 50000     # 50 km search radius

    # est_SK is the estimate and var_SK is the variance
    est_SK, var_SK = gs.Interpolation.okrige(
        Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad)

    expected_est = np.array([443.9, 299.5, 356.6, 389.8, 160.5,  82.2, 333.4, 228.8, 413.2,
                            376., 201.4, 319.4, 286.2, 298.5, 368.8, 399.2, 337., 132.3,
                            305.9, 247.2, 270.5, 115.1, 417.5, 411.4,  32.7])

    expected_var = np.array([6525.9,     0.,  8308.6, 12133.4, 11820.4,  8437.8, 13252.4,
                            11871.6, 19809.1,  8048.9,  6762.9, 20021.6, 15386.6, 10189.8,
                            0.,  6480.3, 12443.7,  5510.,  6256.2, 13197.6,     0.,
                            10845.6, 17350.8,  3980.5,  8585.3])

    # assert
    np.testing.assert_array_almost_equal(est_SK, expected_est, decimal=1)
    np.testing.assert_array_almost_equal(var_SK, expected_var, decimal=1)


def test_simple_kriging():
    """
    Test of simple kriging.
    The test is roughly based on demos/3_Simple_kriging_and_ordinary_kriging.ipynb

    """
    # read demo data
    data_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../demos/data/greenland_test_data.csv')
    df_bed = pd.read_csv(data_file_path)

    # grid data to 100 m resolution and remove coordinates with NaNs
    res = 1000
    df_grid, _, _, _ = gs.Gridding.grid_data(
        df_bed, 'X', 'Y', 'Bed', res)
    df_grid = df_grid[df_grid["Z"].isnull() == False]

    # define coordinate grid
    xmin = np.min(df_grid['X'])
    xmax = np.max(df_grid['X'])     # min and max x values
    ymin = np.min(df_grid['Y'])
    ymax = np.max(df_grid['Y'])     # min and max y values

    Pred_grid_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)

    # set random seed
    np.random.seed(42)

    # pick ("random") points from grid
    index_points = np.random.choice(
        range(len(Pred_grid_xy)), size=25, replace=False)
    Pred_grid_xy = Pred_grid_xy[index_points, :]

    # set variogram parameters
    azimuth = 0
    nugget = 0

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = 19236.
    minor_range = 19236.
    sill = 22399.
    vtype = 'Exponential'

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

    k = 100         # number of neighboring data points used to estimate a given point
    rad = 50000     # 50 km search radius

    # est_SK is the estimate and var_SK is the variance
    est_SK, var_SK = gs.Interpolation.skrige(
        Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad)

    expected_est = np.array([432.3, 299.5, 354.5, 384.8, 166.9,  82.9, 327.7, 230.1, 333.,
                            374.8, 208.7, 287., 296.2, 297.1, 368.8, 391.5, 336.4, 133.,
                            307.3, 249.4, 270.5, 119.2, 369.9, 411.2,  38.1])

    expected_var = np.array([6846.7,     0.,  8367.2, 12287., 12029.1,  8573.8, 13565.,
                            12098.3, 20816.3,  8134.4,  7099.8, 20588.1, 15810.4, 10312.,
                            0.,  6739., 12586.4,  5530.8,  6312., 13469.5,     0.,
                            10969.9, 18156.7,  3992.2,  8701.3])

    # assert
    np.testing.assert_array_almost_equal(est_SK, expected_est, decimal=1)
    np.testing.assert_array_almost_equal(var_SK, expected_var, decimal=1)


def test_sequential_gaussian_simulation_ordinary_kriging():
    """
    This tests the sequential gaussian simulation with ordinary kriging.
    The test is roughly based on demos/4_Sequential_Gaussian_Simulation.ipynb

    """
    # read demo data
    data_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../demos/data/greenland_test_data.csv')
    df_bed = pd.read_csv(data_file_path)

    # Grid and transform data, compute variogram parameters
    # grid data to 100 m resolution and remove coordinates with NaNs
    res = 1000
    df_grid, _, _, _ = gs.Gridding.grid_data(df_bed, 'X', 'Y', 'Bed', res)

    # remove NaNs
    df_grid = df_grid[df_grid["Z"].isnull() == False]

    # Initialize grid
    # define coordinate grid
    xmin = np.min(df_grid['X'])
    xmax = np.max(df_grid['X'])     # min and max x values

    ymin = np.min(df_grid['Y'])
    ymax = np.max(df_grid['Y'])     # min and max y values

    Pred_grid_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)

    # set random seeds
    # this line can be removed, if we decide to use numpy.random.shuffle instead of random.shuffle
    random.seed(42)
    np.random.seed(42)

    # pick ("random") points from grid
    index_points = np.random.choice(
        range(len(Pred_grid_xy)), size=25, replace=False)
    Pred_grid_xy = Pred_grid_xy[index_points, :]

    # Sequential Gaussian simulation
    # set variogram parameters
    azimuth = 0
    nugget = 0
    k = 48         # number of neighboring data points used to estimate a given point
    rad = 50000    # 50 km search radius

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = minor_range = 31852.
    sill = 0.7
    vtype = 'Exponential'

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

    # ordinary kriging
    sim = gs.Interpolation.okrige_sgs(
        Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad)

    # as we set the numpy random seed, the simulation is deterministic and we can compare to the following (rounded) results
    expected_sim = np.array([445.5, 299.5, 358.9, 395.9, 153.5,  78.3, 344.5, 226.1, 432.2,
                             377.1, 202.9, 319.3, 276.9, 296., 368.8, 405.2, 328.2, 134.6,
                             307.8, 252.3, 270.5, 117.9, 425., 411.6,  31.2])

    # assert
    np.testing.assert_array_almost_equal(sim, expected_sim, decimal=1)


def test_sequential_gaussian_simulation_simple_kriging():
    """
    This tests the sequential gaussian simulation with simple kriging.
    The test is roughly based on demos/4_Sequential_Gaussian_Simulation.ipynb

    """
    data_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../demos/data/greenland_test_data.csv')
    df_bed = pd.read_csv(data_file_path)

    # Grid and transform data, compute variogram parameters
    # grid data to 100 m resolution and remove coordinates with NaNs
    res = 1000
    df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(
        df_bed, 'X', 'Y', 'Bed', res)

    # remove NaNs
    df_grid = df_grid[df_grid["Z"].isnull() == False]

    # maximum range distance
    maxlag = 50000
    # num of bins
    n_lags = 70

    # Initialize grid
    # define coordinate grid
    xmin = np.min(df_grid['X'])
    xmax = np.max(df_grid['X'])     # min and max x values

    ymin = np.min(df_grid['Y'])
    ymax = np.max(df_grid['Y'])     # min and max y values

    Pred_grid_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)

    # set random seed
    # this line can be removed, if we decide to use numpy.random.shuffle instead of random.shuffle
    random.seed(42)
    np.random.seed(42)

    # pick ("random") points from grid
    index_points = np.random.choice(
        range(len(Pred_grid_xy)), size=25, replace=False)
    Pred_grid_xy = Pred_grid_xy[index_points, :]

    # Sequential Gaussian simulation
    # set variogram parameters
    azimuth = 0
    nugget = 0
    k = 48         # number of neighboring data points used to estimate a given point
    rad = 50000    # 50 km search radius

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = minor_range = 31852.
    sill = 0.7
    vtype = 'Exponential'

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

    # simple kriging
    sim = gs.Interpolation.skrige_sgs(
        Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad)

    # as we set the numpy random seed, the simulation is deterministic and we can compare to the following (rounded) results
    expected_sim = np.array([440.3, 299.5, 360., 396.7, 152.1,  78.3, 344.5, 225.6, 368.2,
                             378.1, 205.2, 304.5, 294.2, 296.3, 368.8, 400.4, 329.9, 133.9,
                             307.4, 252.2, 270.5, 116.5, 403.3, 411.6,  29.1])

    # assert
    np.testing.assert_array_almost_equal(sim, expected_sim, decimal=1)


if __name__ == '__main__':
    import pytest
    pytest.main()

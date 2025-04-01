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
    rng = np.random.default_rng(42)

    # pick ("random") points from grid
    index_points = rng.choice(
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

    expected_est = np.array([ 446.2, 416.3, -16.3, 293.8, 267.7, 383.5, 7.9, 350.9,
                             78.9, -108.3, 484.1, 385.3, 323.4, 386.4, 365.7, 436.7,
                             421.8, 370.2, 349.3, 170.7, 0.7, 182.5, 385.2, 219.6, 318.8])

    expected_var = np.array([7645.2, 4755.7, 9880.7, 9882.3, 11269.9, 16169.5, 10920.9,
                             11398.8, 0., 12246., 0., 21342.3, 7936.8, 4780., 11525.4,
                             11745.7, 3763.3, 19738.4, 4070.6, 0., 8548.9, 3507.8,
                             21654.5, 4748.4, 3767.])

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
    rng = np.random.default_rng(42)

    # pick ("random") points from grid
    index_points = rng.choice(
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

    expected_est = np.array([444.1, 415.3, -9.3, 296., 269.2, 340.9, 15.9, 335.6, 78.9,
                             -94.3, 484.1, 311.5, 315.4, 386.3, 371.1, 430.5, 421.8, 305.9,
                             349.8, 170.7, 8.4, 182.5, 290.4, 223.6, 318.9])

    expected_var = np.array([7711.6, 4787.7, 10011.4, 10014.5, 11474.8, 17027.6, 11092.1,
                             11777.1, 0., 12449.5, 0., 21794.7, 8183.3, 4832.2, 11771.3,
                             11899.5, 3765.1, 20914.3, 4055.8, 0., 8889.6, 3522.3, 22182.6,
                             4797.8, 3740.1])

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
    # random.seed(42)
    # np.random.seed(42)
    rng = np.random.default_rng(42)

    # pick ("random") points from grid
    index_points = rng.choice(
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
        Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad, rng=rng)

    # as we set the numpy random seed, the simulation is deterministic and we can compare to the following (rounded) results
    expected_sim = np.array([444.9, 415.8, -23.7, 301.2, 263.3, 374.9, -6.9, 346.5,
                             78.9, -106.6, 484.1, 374.8, 321.6, 388.6, 366.8, 440.1,
                             421.6, 367., 349.6, 170.7, -7.7, 182., 379.8, 221.3, 318.3])

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
    # random.seed(42)
    # np.random.seed(42)
    rng = np.random.default_rng(42)

    # pick ("random") points from grid
    index_points = rng.choice(
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
        Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad, rng=rng)

    # as we set the numpy random seed, the simulation is deterministic and we can compare to the following (rounded) results
    expected_sim = np.array([445.6, 416.4, -27.6, 300.8, 263.4, 352.5, -7.1, 341.7,
                             78.9, -110.7, 484.1, 338., 318.1, 388.6, 366.9, 441.4,
                             421.6, 323., 349.9, 170.7, -4.4, 182., 309.3, 220.7, 318.6])

    # assert
    np.testing.assert_array_almost_equal(sim, expected_sim, decimal=1)


if __name__ == '__main__':
    import pytest
    pytest.main()

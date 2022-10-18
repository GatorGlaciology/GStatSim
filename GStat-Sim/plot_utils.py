import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import earthpy.spatial as es

def make_colorbar(fig, im, vmin, vmax, clabel, ax=None):
    if ax is None:
        ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, ticks=np.linspace(vmin, vmax, 11), cax=cax)
    cbar.set_label(clabel, rotation=270, labelpad=15)
    return cbar

def splot2D(df, title, xlabel='X [m]', ylabel='Y [m]', clabel='Bed [m]', x='X', y='Y', c='Bed', vmin=-400, vmax=600, s=0.5):
    fig, ax = plt.subplots(1, figsize=(5,5))
    im = plt.scatter(df[x], df[y], c=df[c], vmin=vmin, vmax=vmax, 
                     marker='.', s=s, cmap='gist_earth')
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.locator_params(nbins=5)

    # make colorbar
    cbar = make_colorbar(fig, im, vmin, vmax, clabel)

    ax.axis('scaled')
    plt.show()
    
def mplot1(Pred_grid_xy, sim, rows, cols, title, xlabel='X [m]', ylabel='Y [m]', 
           clabel='Bed [m]', vmin=-400, vmax=600, hillshade=False):
    x_mat = Pred_grid_xy[:,0].reshape((rows, cols))
    y_mat = Pred_grid_xy[:,1].reshape((rows, cols))
    mat = sim.reshape((rows, cols))

    xmin = Pred_grid_xy[:,0].min(); xmax = Pred_grid_xy[:,0].max()
    ymin = Pred_grid_xy[:,1].min(); ymax = Pred_grid_xy[:,1].max()

    fig, ax = plt.subplots(1, figsize=(5,5))
    im = plt.pcolormesh(x_mat, y_mat, mat, vmin=vmin, vmax=vmax, cmap='gist_earth')
    if hillshade == True:
        hillshade = es.hillshade(mat, azimuth=210, altitude=10)
        plt.pcolormesh(x_mat, y_mat, hillshade, cmap='Greys', alpha=0.1)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.xticks(np.linspace(xmin, xmax, 5))
    plt.yticks(np.linspace(ymin, ymax, 5))

    # make colorbar
    if hillshade is False:
        cbar = make_colorbar(fig, im, vmin, vmax, clabel)
    
    ax.axis('scaled')
    plt.show()
    
def mplot2_std(Pred_grid_xy, pred, std, rows, cols, title1, title2):    
    x_mat = Pred_grid_xy[:,0].reshape((rows, cols))
    y_mat = Pred_grid_xy[:,1].reshape((rows, cols))
    pred_mat = pred.reshape((rows, cols))
    std_mat = std.reshape((rows, cols))
    
    xmin = Pred_grid_xy[:,0].min(); xmax = Pred_grid_xy[:,0].max()
    ymin = Pred_grid_xy[:,1].min(); ymax = Pred_grid_xy[:,1].max()

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10,5))

    im = ax1.pcolormesh(x_mat, y_mat, pred_mat, vmin=-400, vmax=600, cmap='gist_earth')
    ax1.set_xticks(np.linspace(xmin, xmax, 5))
    ax1.set_yticks(np.linspace(ymin, ymax, 5))
    ax1.set_title(title1)
    ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]')
    ax1.axis('scaled')

    # make colorbar
    cbar = make_colorbar(fig, im, -400, 600, 'Bed [m]', ax=ax1)

    im = ax2.pcolormesh(x_mat, y_mat, std_mat, vmin=0, vmax=150, cmap='gist_earth')
    ax2.set_xticks(np.linspace(xmin, xmax, 5))
    ax2.set_yticks(np.linspace(ymin, ymax, 5))
    ax2.set_title(title2)
    ax2.set_xlabel('X [m]'); ax2.set_ylabel('Y [m]')
    ax2.axis('scaled')

    # make colorbar
    cbar = make_colorbar(fig, im, 0, 150, 'Bed [m]', ax=ax2)

    plt.tight_layout()
    plt.show()
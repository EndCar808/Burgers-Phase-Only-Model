#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros

######################
##  Library Imports
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file

import matplotlib.pyplot as plt
# plt.style.use('seaborn-talk')
mpl.rcParams['figure.figsize'] = [16, 9]
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.size']   = 20
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
mpl.rcParams['lines.markersize'] = 6
# mpl.rcParams['lines.linewidth'] = 1.25
# mpl.rcParams['lines.markersize'] = 6
from scipy.io import FortranFile
kr='float64'
ki='int64'
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import h5py
import sys
import os
import time as TIME
import multiprocessing as mprocs
from threading import Thread
from subprocess import Popen, PIPE
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm 
import numpy as np 
np.set_printoptions(threshold=sys.maxsize)
from numba import jit, njit


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from functions import open_file
from stats_functions import normalized_pdf, percentiles


#######################
##   Function Defs   ##
#######################
def percentiles_over_alpha(N, alpha, k0, beta, iters, trans, input_dir):

    ## Allocate memory    
    incr_percentiles_left  = np.zeros((len(alpha)))
    incr_percentiles_right = np.zeros((len(alpha)))

    for i, a in enumerate(alpha):

        ## Open file
        file = open_file(a, N, k0, beta, u0, iters, trans, input_dir)

        ## Read in data
        small_incr_counts     = file['VelInc[0]_BinCounts'][:]
        small_incr_bins_edges = file['VelInc[0]_BinEdges'][:]

        incr_percentiles_left[i], incr_percentiles_right[i] = percentiles(small_incr_counts, small_incr_bins_edges, 5.0, pdf = True)


    return incr_percentiles_left, incr_percentiles_right


def stat_over_alpha(N, alpha, k0, beta, iters, trans, input_dir):

    ## Allocate arrays
    small_skew = np.zeros((len(alpha)))
    small_kurt = np.zeros((len(alpha)))

    ## Loop over alpha
    for i, a in enumerate(alpha):

        ## Open file
        file = open_file(a, N, k0, beta, u0, iters, trans, input_dir)

        ## Read in data
        small_skew[i] = file["VelIncStats"][0, 2]
        small_kurt[i] = file["VelIncStats"][0, 3]

    return small_skew, small_kurt



def pdf_over_alpha(N, alpha, k0, beta, iters, trans, input_dir, nbins, grad = False):

    ## Allocate arrays
    small_pdf     = np.zeros((len(alpha)))
    small_bin_pts = np.zeros((len(alpha)))
    large_pdf     = np.zeros((len(alpha)))
    large_bin_pts = np.zeros((len(alpha)))
    if grad == True:
        grad_pdf     = np.zeros((len(alpha)))
        grad_bin_pts = np.zeros((len(alpha)))

    ## Loop over alpha
    for i, a in enumerate(alpha): 

        ## Open file
        file = open_file(a, N, k0, beta, u0, iters, trans, input_dir)

        ## Read in data
        small_incr_counts     = file['VelInc[0]_BinCounts'][:]
        small_incr_bins_edges = file['VelInc[0]_BinEdges'][:]
        large_incr_counts     = file['VelInc[1]_BinCounts'][:]
        large_incr_bins_edges = file['VelInc[1]_BinEdges'][:]
   
        ## Small scale pdf
        small_pdf[i, :], small_bin_pts[i, :], _ = normalized_pdf(small_incr_counts, small_incr_bins_edges)

        ## Large scale pdf
        large_pdf[i, :], large_bin_pts[i, :], _ = normalized_pdf(large_incr_counts, large_incr_bins_edges)

        ## Gradient pdf
        if grad == True:
            grad_counts     = file['VelGrad_BinCounts'][:]
            grad_bins_edges = file['VelGrad_BinEdges'][:]

            grad_pdf[i, :], grad_bin_pts[i, :], _ = normalized_pdf(grad_counts, grad_bins_edges)


    if grad == True:
        small_pdf, small_bin_pts, large_pdf, large_bin_pts, grad_pdf, grad_bin_pts
    else:
        small_pdf, small_bin_pts, large_pdf, large_bin_pts




######################
##  Main
######################
if __name__ == '__main__':




    #########################
    ##  Get Input Parameters
    #########################
    k0    = 1
    alpha = np.arange(0.0, 2.51, 0.025)
    beta  = 0.0
    iters = int(2e8)
    trans = int(1e6)
    N     = int(2048)
    u0    = "RANDOM"

    num_osc = int(N / 2 + 1)
    kmax    = num_osc - 1
    kmin    = k0 + 1
    num_obs = N * iters



    ######################
    ##  Input & Output Dir
    ######################
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Transfer_Report"



    ######################
    ##  Open Input File
    ######################
    ## Get the intermittency stat looping over alpha
    small_skew, small_kurt = stat_over_alpha(N, alpha, k0, beta, iters, trans, input_dir)

    ## Create figure
    fig = plt.figure(figsize = (12, 7), tight_layout=False)
    gs  = GridSpec(1, 1)
    ax  = fig.add_subplot(gs[0, 0])
    
    ax.plot(alpha, small_skew)
    ax.plot(alpha, small_kurt)
    ax.axhline(y = 0, xmin = alpha[0], xmax = alpha[-1], linestyle = '--', color = 'black')
    ax.set_xlim(alpha[0], alpha[-1])
    ax.set_xlabel(r"$\alpha$")
    # ax.set_yscale('log')
    ax.legend([r"Skewness", r"Kurtosis"])

    plt.savefig(output_dir + "/SKEW_KURT_N[{}]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(N, beta, k0, iters), format='png', dpi = 400)  
    plt.close()




    ## Create Gaussian
    x     = np.linspace(-200, 200, num = 500001)
    Gauss = np.exp(-x**2 * 0.5) / np.sqrt(2.0 * np.pi)
    dx = x[1] - x[0]

    ## Compute the percentiles for the Gaussian
    gauss_percentile_left, gauss_percentile_right = percentiles(Gauss, x, 5.0)
    
    ## Get the percentiles looping over alpha
    incr_percentiles_left, incr_percentiles_right = percentiles_over_alpha(N, alpha, k0, beta, iters, trans, input_dir)


    ## Plot
    fig = plt.figure(figsize = (12, 7), tight_layout=False)
    gs  = GridSpec(1, 1)
    ax  = fig.add_subplot(gs[0, 0])
    ax.plot(alpha, incr_percentiles_left / gauss_percentile_left)
    ax.plot(alpha, incr_percentiles_right / gauss_percentile_right)
    ax.set_xlim(alpha[0], alpha[-1])
    ax.set_xlabel(r"$\alpha$")
    # ax.set_yscale('log')
    ax.legend([r"Left", r"Right"])
    plt.savefig(output_dir + "/PERCENTILES_N[{}]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(N, beta, k0, iters), format='png', dpi = 400)  
    plt.close()





    #######################
    ##    Paper Plot     ##
    #######################
    ## Create figure
    fig = plt.figure(figsize = (7.0, 5.4), tight_layout=False)
    gs  = GridSpec(4, 3, hspace = 0.6, wspace = 0.3)

    labelsize = 8
    ticksize = 8
    font_size = 10
    
    ## Create pdf axes
    ax_pdf = []
    for i in range(2):
        for j in range(3):
            ax_pdf.append(fig.add_subplot(gs[i, j]))

    # ## Create percentile and stats axes
    # ax_percent = fig.add_subplot(gs[2:4, :])
    ax_stats   = fig.add_subplot(gs[2:4, :])

    grad = False

    ## Create gaussian
    x        = np.linspace(-5, 5, 10000)
    gaussian = np.exp( -x**2 * 0.5) / np.sqrt(2 * np.pi) 

    ## Fill the PDF axes
    for i, a in enumerate([0.0, 0.5, 1.0, 1.2, 1.5, 2.0]): #enumerate([0.0, 0.5, 1.0, 1.2, 1.5, 1.7, 2.0, 2.2]): 

        ## Open file
        file = open_file(a, N, k0, beta, u0, iters, trans, input_dir)

        ## Read in data
        small_incr_counts     = file['VelInc[0]_BinCounts'][:]
        small_incr_bins_edges = file['VelInc[0]_BinEdges'][:]
        large_incr_counts     = file['VelInc[1]_BinCounts'][:]
        large_incr_bins_edges = file['VelInc[1]_BinEdges'][:]
        
        ## Small scale pdf
        small_pdf, small_bin_pts, _ = normalized_pdf(small_incr_counts, small_incr_bins_edges)

        ## Large scale pdf
        large_pdf, large_bin_pts, _ = normalized_pdf(large_incr_counts, large_incr_bins_edges)

        ## Gradient pdf
        if grad == True:
            grad_counts     = file['VelGrad_BinCounts'][:]
            grad_bins_edges = file['VelGrad_BinEdges'][:]

            grad_pdf, grad_bin_pts, _ = normalized_pdf(grad_counts, grad_bins_edges)


        ## Plot the pdf data
        ax_pdf[i].plot(small_bin_pts, small_pdf)
        ax_pdf[i].plot(large_bin_pts, large_pdf)
        if grad == True:
            ax_pdf[i].plot(grad_bin_pts, grad_pdf)
        ax_pdf[i].plot(x, gaussian, '--', color = 'black')
        ax_pdf[i].set_yscale('log')
        ax_pdf[i].tick_params(axis = 'x', labelsize = font_size)
        ax_pdf[i].tick_params(axis = 'y', labelsize = font_size)
        ax_pdf[i].set_ylim(bottom = 1.5e-7)


    for i in range(2):
        ax_pdf[i * 3 + 0].set_ylabel(r"PDF")

    for i in range(3):
        ax_pdf[3 + i].set_xlabel(r"$\delta u_r / \sigma$", labelpad = 0.8)


    
    plt.gcf().text(x = 0.05, y = 0.88, s = r"a)", fontsize = font_size)
    plt.gcf().text(x = 0.05, y = 0.67, s = r"d)", fontsize = font_size)
    plt.gcf().text(x = 0.37, y = 0.88, s = r"b)", fontsize = font_size)
    plt.gcf().text(x = 0.37, y = 0.67, s = r"e)", fontsize = font_size)
    plt.gcf().text(x = 0.625, y = 0.88, s = r"c)", fontsize = font_size)
    plt.gcf().text(x = 0.625, y = 0.67, s = r"f)", fontsize = font_size)

    ax_pdf[0].text(x = -6, y = 1e-1, s = r"$\alpha = {:0.2f}$".format(0.0), fontsize = font_size - 2)
    ax_pdf[1].text(x = -12, y = 1e-1, s = r"$\alpha = {:0.2f}$".format(0.5), fontsize = font_size - 2)
    ax_pdf[2].text(x = -30, y = 1e-1, s = r"$\alpha = {:0.2f}$".format(1.0), fontsize = font_size - 2)
    ax_pdf[3].text(x = -30, y = 1e-1, s = r"$\alpha = {:0.2f}$".format(1.2), fontsize = font_size - 2)
    ax_pdf[4].text(x = -25, y = 1e-1, s = r"$\alpha = {:0.2f}$".format(1.5), fontsize = font_size - 2)
    ax_pdf[5].text(x = -9, y = 1e-1, s = r"$\alpha = {:0.2f}$".format(2.0), fontsize = font_size - 2)


    ## Fill the stats axes
    small_skew, small_kurt = stat_over_alpha(N, alpha, k0, beta, iters, trans, input_dir)
    skew = ax_stats.plot(alpha, np.absolute(small_skew), color = 'c')
    kurt = ax_stats.plot(alpha, small_kurt, color = 'm')
    ax_stats.axhline(y = 0, xmin = alpha[0], xmax = alpha[-1], linestyle = '--', color = 'black')
    ax_stats.set_xlim(alpha[0], alpha[-1])
    ax_stats.set_xlabel(r"$\alpha$")
    ax_stats.tick_params(axis = 'x', labelsize = font_size)
    ax_stats.tick_params(axis = 'y', labelsize = font_size)
    # ax_stats.set_yscale('log')
    ax_stats.legend([r"$|\mathcal{S}|$", r"$\mathcal{K}$"])
    axins = inset_axes(ax_stats, width = "30%", height = "40%", loc = 2, borderpad = 2.75)
    axins.set_xlim([alpha[0],alpha[-1]])
    axins.plot(alpha, np.absolute(small_skew), color = 'c')
    axins.plot(alpha, small_kurt, color = 'm')
    axins.set_yscale('log')
    axins.tick_params(axis = 'x', labelsize = font_size)
    axins.tick_params(axis = 'y', labelsize = font_size)
    plt.gcf().text(x = 0.05, y = 0.475, s = r"g)", fontsize = font_size)
    
    # ## Fill the percentiles axes
    # x     = np.linspace(-200, 200, num = 500001)
    # Gauss = np.exp(-x**2 * 0.5) / np.sqrt(2.0 * np.pi)
    # dx = x[1] - x[0]

    # ## Compute the percentiles for the Gaussian
    # gauss_percentile_left, gauss_percentile_right = percentiles(Gauss, x, 5.0)
    
    # ## Get the percentiles looping over alpha
    # incr_percentiles_left, incr_percentiles_right = percentiles_over_alpha(N, alpha, k0, beta, iters, trans, input_dir)


    # ## Plot
    # ax_percent.plot(alpha, incr_percentiles_left / gauss_percentile_left)
    # ax_percent.plot(alpha, incr_percentiles_right / gauss_percentile_right)
    # ax_percent.set_xlim(alpha[0], alpha[-1])
    # ax_percent.set_xlabel(r"$\alpha$")
    # # ax_percent.set_yscale('log')
    # ax_percent.legend([r"Left", r"Right"])


    plt.savefig(output_dir + "/STATS_N[{}]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].pdf".format(N, beta, k0, iters), bbox_inches='tight')  
    plt.close()
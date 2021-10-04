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
mpl.rcParams['figure.figsize']    = [5, 3]
# mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['text.usetex']       = True
mpl.rcParams['font.family']       = 'serif'
mpl.rcParams['font.serif']        = 'Computer Modern Roman'
mpl.rcParams['axes.labelsize']    = 6
mpl.rcParams['font.size']         = 8
# mpl.rcParams['lines.linewidth']   = 0.75
# mpl.rcParams['lines.markersize']  = 6
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import h5py
import sys
import os
import time as TIME
import pandas as pd
# import multiprocessing as mprocs
# from threading import Thread
# from subprocess import Popen, PIPE
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import zip_longest
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm 
import numpy as np
from numba import jit, njit, prange
from scipy.stats import linregress
import itertools 
from functions import open_file, compute_triads


if __name__ == '__main__':

    #########################
    ##  Get Input Parameters
    #########################
    k0    = 1
    N     = [1024, 2048, 4096, 8192, 16384] # , 32768]
    alpha = [0.0, 0.5, 1.0, 1.2, 1.5, 1.7, 2.0, 2.2, 2.5] # [1.2, 1.5, 1.7, 2.0, 2.2, 2.5]
    beta  = 0.0
    u0    = 'RANDOM'
    iters = int(5e6)
    trans = 100000000

  
    
    ######################
    ##  Input & Output Dir
    ######################
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/PhaseOrder"


    ######################
    ##  Plot the data
    ######################
    fig = plt.figure(figsize = (16, 9), tight_layout = False)
    xg = 3
    yg = 3
    gs  = GridSpec(xg, yg)  

    ax = []
    for i in range(xg):
        for j in range(yg):
            ax.append(fig.add_subplot(gs[i, j]))


    for n in N:

        print("n = {}".format(n))

        kmin = k0 + 1
        kmax = int(n / 2)
        num_osc = kmax + 1

        indx = 0

        for a in alpha:

            ## Open file
            data_short = open_file(a, n, k0, beta, u0, iters, trans, input_dir)
            data_long  = open_file(a, n, k0, beta, u0, int(iters * 2), trans, input_dir)

            ## Read in the R_k data
            R_k_short = data_short['R_k_avg'][:]
            R_k_long  = data_long['R_k_avg'][:]

            ## Plot data
            ax[indx].plot(np.arange(kmin, num_osc), R_k_short[kmin:], '-')
            ax[indx].plot(np.arange(kmin, num_osc), R_k_long[kmin:], '--')
            ax[indx].legend([r"T / 2", r"T"])
            ax[indx].set_title(r"$\alpha = {}$".format(a))

            indx += 1
            indx = np.mod(indx, xg * yg)

        ## Add axes labels
        for i in range(3):
            ax[(yg - 1) * 3 + i].set_xlabel(r"$k$")
        for i in range(3):
            ax[i * 3 + 0].set_ylabel(r"$R_k$")
            ax[i * 3 + 0].set_yticks([0, 0.5, 1])
            ax[i * 3 + 0].set_yticklabels([0, 0.5, 1.0])

        ## Add text to show which alpha is plotted
        for i, a in enumerate(alpha):
            # if i <= 2:
            #     ax[i].text(1000, 0.8, r"$\alpha = {:0.2f}$".format(a))
            # else:
            #     ax[i].text(1000, 0.25, r"$\alpha = {:0.2f}$".format(a))
            ax[i].set_ylim(0, 1)
            ax[i].set_xlim(k0 + 1, int(n / 2))

        ## Save figure
        plt.savefig(output_dir + "/R_k_timing_N[{}]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(n, beta, k0, iters, u0), bbox_inches='tight')
        
        ## Clear axes
        for i in range(xg * yg):
            ax[i].cla()
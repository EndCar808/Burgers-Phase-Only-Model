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
# mpl.rcParams['axes.labelsize']    = 6
# mpl.rcParams['font.size']         = 8
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
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm 
import numpy as np
from numba import jit, njit, prange
import itertools 
# from scipy.stats import linregress
# from numpy.polynomial.polynomial import polyfit

from functions import open_file



def normalize_time_order(order, kmin):

    ## Get params
    num_t_steps = order.shape[0]
    num_osc     = order.shape[1]

    for t in range(num_t_steps):
        for k in range(kmin, num_osc):
            order[t, k] /= np.absolute(order[t, k])

    return order



def compute_time_order(order, kmin):

    ## Get params
    num_t_steps = order.shape[0]
    num_osc     = order.shape[1]

    tmp_time_order = np.ones((num_osc, ), dtype = 'complex128') 
    time_order      = np.zeros((num_t_steps, num_osc), dtype = 'complex128')

    t_count = 1
    for t in range(num_t_steps):
        for k in range(kmin, num_osc):
            tmp_time_order[k] += np.exp(1j * np.angle(order[t, k]))
            time_order[t, k] = tmp_time_order[k] / t_count
        t_count += 1

    return time_order


if __name__ == '__main__':
    #########################
    ##  Get Input Parameters
    #########################
    k0    = 1
    N     = [1024]
    alpha = [1.0]
    beta  = 0.0
    u0    = 'RANDOM'
    iters = int(1e5)
    trans = int(1e5)
    
    ######################
    ##  Input & Output Dir
    ######################
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/PhaseOrder/Investigate"


    ## Open file
    file = open_file(alpha[0], N[0], k0, beta, u0, iters, trans, input_dir)


    ## Read in data
    order = file["PhaseShiftScaleOrderParam"][:, :]

    ## Compute time order
    time_order = compute_time_order(order, k0 + 1)

    ## Compute normed time order
    normed_order = normalize_time_order(order, k0 + 1)

    ## Compute time order
    normed_time_order = compute_time_order(normed_order, k0 + 1)


    ## Compute R_k
    r_k = np.absolute(time_order[-1, :])
    normed_r_k = np.absolute(normed_time_order[-1, :])

    fig = plt.figure(figsize = (7, 2.4), tight_layout = False)
    plt.plot(r_k)
    plt.plot(normed_r_k)
    plt.legend([r"$R_k$", r"Normed $R_k$"])

    plt.savefig(output_dir + "/R_k_vs_NORMED_R_k_FULL_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()


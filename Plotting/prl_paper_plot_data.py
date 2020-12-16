######################
##	Library Imports ##
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize']    = [16, 9]
mpl.rcParams['text.usetex']       = True
mpl.rcParams['font.family']       = 'serif'
mpl.rcParams['font.size']         = 22
mpl.rcParams['font.serif']        = 'Computer Modern Roman'
mpl.rcParams['lines.linewidth']   = 1.25
mpl.rcParams['lines.markersize']  = 6
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import h5py
import sys
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm 
import numpy as np
import itertools




######################
##	     MAIN       ##
######################
if __name__ == '__main__':

    ## ------ Parameters
    N     = [128, 256, 512, 1024]
    alpha = np.arange(0.0, 2.55, 0.05)
    k0    = 1
    beta  = 0.0
    iters = 400000
    m_end = 8000
    m_itr = 50
    trans = 0
    u0    = "RANDOM"


    ## ------ Input and Output Dirs
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/Paper"
    output_dir = input_dir

    ## ------ Open Data File
    PaperData = h5py.File(input_dir + "/PaperData.hdf5", "r")

    ## ------ Read in Data
    quartile         = PaperData['Quartiles'][:, :]
    deg_of_freedom   = PaperData['DOF'][:]
    kaplan_york_dim  = PaperData['KaplanYorke'][:, :]
    max_mean_k_enorm = PaperData['MaxMeankENorm'][:, :]

    ## ------ Plot Data
    fig = plt.figure(figsize = (12, 18), tight_layout = False)
    gs  = GridSpec(3, 1, hspace = 0.1)

    mpl.rcParams['font.size'] = 23
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['axes.linewidth']    = 2
    xlabel_size = 26
    ylabel_size = 26

    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(len(N)):
        plt.plot(alpha[:], kaplan_york_dim[i, :] / deg_of_freedom[i], '.-')
    ax1.set_xlim(alpha[0], alpha[-1])
    ax1.set_ylim(0, 1)
    ax1.set_xticklabels([])
    ax1.set_ylabel(r"$\frac{\mathcal{D}_{KY}}{ N - 1}$", fontsize = ylabel_size)

    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(len(N)):
        ax2.plot(alpha, max_mean_k_enorm[i, :] / deg_of_freedom[i], '.-')
        if i == len(N) - 1:
            plt.fill_between(alpha, quartile[0, i, :] / deg_of_freedom[i], quartile[1, i, :] / deg_of_freedom[i], alpha = 0.2, color = 'red')
    ax2.set_xlim(alpha[0], alpha[-1])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel(r"$\frac{\overline{K}}{N - 1}$", fontsize = ylabel_size)
    ax2.set_xlabel(r"$\alpha$", fontsize = xlabel_size)
    ax2.set_xlim(alpha[0], alpha[-1])

    ax1.text(-0.25, 1, r"(a)")
    ax2.text(-0.25, 1, r"(b)")
    ax1.legend([r"$N = {}$".format(int(n / 2)) for n in N], bbox_to_anchor = (0.001, 1.0, 1, 0.2), loc="lower left", mode = "expand", ncol = len(N))

    plt.savefig(output_dir + "/PAPER_WNORM_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()
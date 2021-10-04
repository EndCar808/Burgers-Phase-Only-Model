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
    N     = [512, 1024] # , 2048, 4096, 8192, 16384, 32768]
    alpha = [0.0, 0.5, 1.0, 1.2, 1.5, 1.7, 2.0, 2.2, 2.5] # [1.2, 1.5, 1.7, 2.0, 2.2, 2.5]
    beta  = 0.0
    u0    = 'RANDOM'
    iters = int(1e6)
    trans = 100000
    
    ######################
    ##  Input & Output Dir
    ######################
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    




    for n in N:

        output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/LimitingDist/N[{}]".format(n)

        for a in alpha:

            # Open file
            HDFfileData = open_file(a, n, k0, beta, u0, iters, trans, input_dir)

            # Read in phases
            phi = HDFfileData['Phases'][:, :]

            # System variables
            num_t_steps = phi.shape[0]
            num_osc     = phi.shape[1]
            kmin        = k0 + 1
            kmax        = num_osc - 1
            dof         = int(n / 2 - k0)
            
        
            # ####################################################
            # ## Plot distributions
            # ####################################################
            fig = plt.figure(figsize = (16, 9), tight_layout = False)
            xg = 4
            yg = 4
            gs  = GridSpec(xg, yg, hspace = 0.4, wspace = 0.4)  

            ax = []
            for i in range(xg):
                for j in range(yg):
                    ax.append(fig.add_subplot(gs[i, j]))

            index = 0
            count = 1

            for k in range(kmin, num_osc):
               
                # Compute the pdf
                counts, bin_edges = np.histogram(np.mod(phi[:, k], 2 * np.pi), bins = 1000, density = False)
                bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
                bin_width   = bin_edges[1] - bin_edges[0]
                pdf = counts / (np.sum(counts) * bin_width)

                ax[index].plot(bin_centres, pdf)
                ax[index].set_xlabel(r"$\phi_{k}$")
                ax[index].set_title(r"$k = {}$".format(k))
                ax[index].set_ylabel(r"PDF")
                ax[index].set_yscale('log')
                ax[index].set_xticks([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
                ax[index].set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])

                if index == 15:
                    plt.savefig(output_dir + "/Alpha[{:0.3f}]".format(a) + "/Marginal" + "/Phi_k_GROUPED[{}]_PDF_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(count, n, a, beta, k0, iters, u0), bbox_inches='tight') 
                    count += 1
                    for i in range(xg * yg):
                            ax[i].cla()

                index += 1 
                index = np.mod(index, 16)
            plt.close()
            print("Phi_k => [Finished]")

            # ####################################################
            # ## Plot Marginal Triad distributions
            # ####################################################
            # compute the triads
            triads, _, _ = compute_triads(phi, kmin, kmax)

            fig = plt.figure(figsize = (16, 9), tight_layout = False)
            xg = 4
            yg = 4
            gs = GridSpec(xg, yg, hspace = 0.6, wspace = 0.4) 
            ax = []
            for i in range(xg):
                for j in range(yg):
                    ax.append(fig.add_subplot(gs[i, j]))

            indx = 0
            cnt  = 1

            for k in range(kmin, kmax + 1):
                for k1 in range(kmin, int(k/2) + 1):
                    counts, bin_edges = np.histogram(np.mod(triads[k - kmin, k1 - kmin, :], 2.0 * np.pi), bins = 1000, density = False)
                    bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
                    bin_width   = bin_edges[1] - bin_edges[0]
                    pdf_triads  = counts / (np.sum(counts) * bin_width)

                    ax[indx].plot(bin_centres, pdf_triads)
                    ax[indx].set_xlabel(r"$\varphi_{k_1, k - k_1}$")
                    ax[indx].set_title(r"$k_1 = {}, k - k_1 = {}$".format(k1, k - k1))
                    ax[indx].set_ylabel(r"PDF")
                    ax[indx].set_yscale('log')
                    ax[indx].set_xticks([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
                    ax[indx].set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
                    
                    if indx == 15:
                        plt.savefig(output_dir + "/Alpha[{:0.3f}]".format(a) + "/Marginal" + "/Triads_PDF_GROUP[{}]_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(cnt, n, a, beta, k0, iters, u0), bbox_inches='tight') 
                        cnt += 1
                        for i in range(xg * yg):
                            ax[i].cla()

                    indx += 1
                    indx = np.mod(indx, 16)
            plt.close()
            print("Triads => [Finished]")
           

            ####################################################
            ## Plot Joint distributions
            ####################################################
            ## Set plotting parameters
            mpl.rcParams['axes.labelsize'] = 8
            mpl.rcParams['font.size']      = 9

            ## Create the figures and axes for the grouped plots
            fig  = plt.figure(num = 1, figsize = (16, 9), tight_layout = False)
            xg   = 4
            yg   = 4
            gs   = GridSpec(xg, yg, hspace = 0.5, wspace = 0.05) 
            ax   = []
            cbax = []
            for i in range(xg):
                for j in range(yg):
                    ax.append(fig.add_subplot(gs[i, j]))
                    div  = make_axes_locatable(ax[i * yg + j])
                    cbax.append(div.append_axes("right", size = "10%", pad = 0.05))

           
            ## Index counters
            indx = 0
            cnt  = 1

            ## Loop throught triad wavenumbers
            for k in range(kmin, kmax + 1):
                for k1 in range(kmin, int(k /2) + 1):

                    ## Compute histogram
                    H, xedges, yedges = np.histogram2d(np.mod(phi[:, k1], 2 * np.pi), np.mod(phi[:, k - k1], 2 * np.pi), bins = 1000, density = False)
                    x_width = xedges[1] - xedges[0]
                    y_width = yedges[1] - yedges[0]
                    pdf2d   = H / (np.sum(H) * (x_width * y_width))
                    
                    ## Plot joint pdf
                    im = ax[indx].imshow(pdf2d, extent = [0, 2 * np.pi, 0, 2 * np.pi], cmap = cm.jet, norm = mpl.colors.LogNorm())
                    cb = plt.colorbar(im, cax = cbax[indx])
                    ax[indx].set_xticks([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
                    ax[indx].set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
                    ax[indx].set_yticks([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
                    ax[indx].set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
                    ax[indx].set_xlabel(r"$\phi_{k_1}$")
                    ax[indx].set_ylabel(r"$\phi_{k - k_1}$")
                    ax[indx].set_title(r"$k = {}, k_1 = {} \quad-\quad k-k_1 = {} $".format(k, k1, k - k1))                

                    ## Save and clear figure when it is full
                    if indx == 15:
                        plt.savefig(output_dir + "/Alpha[{:0.3f}]".format(a) + "/Joint" +  "/Phi_k_JointPDF_GROUPED[{}]_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(cnt, n, a, beta, k0, iters, u0), bbox_inches='tight') 
                        cnt += 1
                        for i in range(xg * yg):
                            ax[i].cla()
                            cbax[i].cla()

                    ## Update the counters
                    indx += 1
                    indx = np.mod(indx, 16)

            plt.close()
            print("Joint Grouped => [Finished]")
            



            ## Set plotting parameters
            mpl.rcParams['axes.labelsize'] = 10
            mpl.rcParams['font.size']      = 11


            ## Create the figures and axes for the individual plots
            fig1 = plt.figure(num = 2, figsize = (12, 7), tight_layout = False)
            ax1  = plt.gca()
            div1  = make_axes_locatable(ax1)
            cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)


            ## Loop throught triad wavenumbers
            for k in range(kmin, kmax + 1):
                for k1 in range(kmin, int(k /2) + 1):

                    ## Compute histogram
                    H, xedges, yedges = np.histogram2d(np.mod(phi[:, k1], 2 * np.pi), np.mod(phi[:, k - k1], 2 * np.pi), bins = 1000, density = False)
                    x_width = xedges[1] - xedges[0]
                    y_width = yedges[1] - yedges[0]
                    pdf2d   = H / (np.sum(H) * (x_width * y_width))

                    ## Plot the individual joint pdf figure
                    im1 = ax1.imshow(pdf2d, extent = [0, 2 * np.pi, 0, 2 * np.pi], cmap = cm.jet, norm = mpl.colors.LogNorm())
                    cb1 = plt.colorbar(im1, cax = cbax1)
                    ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
                    ax1.set_yticks([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
                    ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
                    ax1.set_xlabel(r"$\phi_{k_1}$")
                    ax1.set_ylabel(r"$\phi_{k - k_1}$")
                    ax1.set_title(r"$k = {}, k_1 = {}  \qquad k = {}, k - k_1 = {}$".format(k, k1, k, k - k1))
                    plt.savefig(output_dir + "/Alpha[{:0.3f}]".format(a) + "/Joint" +  "/Phi_k_Log_JointPDF_kk1[{}, {}]_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(k, k - k1, N, a, beta, k0, iters, u0), bbox_inches='tight') 
                    ax1.cla()
                    cbax1.cla()

            plt.close()
            print("Joint Individual => [Finished]")
        
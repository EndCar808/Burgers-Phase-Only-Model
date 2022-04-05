#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros

######################
##  Library Imports
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file
# mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
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
from numba import jit, njit
from matplotlib import pyplot as plt 
from basic_units import radians


# @njit
# def window_average():



######################
##  Main
######################
if __name__ == '__main__':

    #########################
    ##  Get Input Parameters
    #########################
    if (len(sys.argv) != 8):
        print("No Input Provided, Error.\nProvide: \nk0\nAlpha\nBeta\nIterations\nTransient Iterations\nN\nu0\n")
        sys.exit()
    else: 
        k0    = int(sys.argv[1])
        alpha = float(sys.argv[2])
        beta  = float(sys.argv[3])
        iters = int(sys.argv[4])
        trans = int(sys.argv[5])
        N     = int(sys.argv[6])
        u0    = str(sys.argv[7])

    filename = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(N, k0, alpha, beta, u0, iters, trans)

    num_osc = int(N / 2 + 1)
    kmax    = num_osc - 1
    kmin    = k0 + 1
    num_obs = N * iters

    ######################
    ##  Input & Output Dir
    ######################
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Adler/" + "N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{}]_TRANS[{}]".format(N, k0, alpha, beta, u0, iters, trans)
    print("Input Dir: {}".format(input_dir + filename))

    if os.path.isdir(output_dir) != True:
        print("Creating Output folder...")
        os.mkdir(output_dir)
    print("Output Dir: {}".format(output_dir))


    ######################
    ##  Open Input File
    ######################
    HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')


    ######################
    ##  Read in Data
    ######################
    a_k               = HDFfileData["Amps"][:] # np.ones((num_osc, ))
    time              = HDFfileData["Time"][:]
    R_k_avg           = HDFfileData["R_k_avg"][:]
    scale_order_param = HDFfileData["PhaseShiftScaleOrderParam"][:, :]
    sin_theta_k       = HDFfileData["SinTheta_k"][:, :]
    Phi_k_dot         = HDFfileData["Phi_k_dot"][:, :]

    ######################
    ##  Plot Data
    ######################
    plt.figure()
    plt.plot(R_k_avg)
    plt.xscale('log')
    plt.savefig(output_dir + "/R_k_avg.png")
    plt.close()

    k_indx = [9, 10, int(kmax / 2), int(kmax - 50), int(kmax - 20), int(kmax - 10), kmax]

    for k in k_indx:
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize = (16, 9), gridspec_kw = dict(hspace = 0.2, wspace = 0.3))

        ## Compute theta_k
        theta_k = np.mod(np.angle(scale_order_param[:, k]) + 2.0 * np.pi, 2.0 * np.pi)
        P_k = np.absolute(scale_order_param[:, k]) 
        F_k = k / (2.0 * a_k[k]) * P_k[:]

        ## Plot t series
        ax[0, 0].plot(time, theta_k, yunits = radians)
        ax[0, 0].set_xlabel("t")
        ax[0, 0].set_ylabel(r"$\theta_k$")

        ## Plot histogram
        counts, bin_edges = np.histogram(theta_k, bins = 500)
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
        bin_width = bin_edges[1] - bin_edges[0]
        pdf = counts / (np.sum(counts) * bin_width)
        ax[0, 1].plot(bin_centres, pdf, xunits = radians)
        ax[0, 1].set_xlabel(r"$\theta_k$")
        ax[0, 1].set_ylabel(r"PDF")
        ax[0, 1].set_yscale('log')

        ## Plot histogram of P_k and F_k and P_k over time
        div3   = make_axes_locatable(ax[1, 0])
        axtop3 = div3.append_axes("top", size = "100%", pad = 0.4)
        counts, bin_edges = np.histogram(P_k, bins = 500)
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
        bin_width = bin_edges[1] - bin_edges[0]
        pdf = counts / (np.sum(counts) * bin_width)
        axtop3.plot(bin_centres, pdf)
        axtop3.set_xlabel(r"P_k")
        axtop3.set_yscale('log')
        ax[1, 0].plot(time, P_k, yunits = radians)
        ax[1, 0].plot(time, F_k, yunits = radians)
        ax[1, 0].set_xlabel("t")
        ax[1, 0].set_ylabel(r"Amplitude Data")
        ax[1, 0].legend([r"$P_k$", r"$F_k$"])
        ax[1, 0].set_yscale('log')

        ## Plot sin(theta_k)
        ax[0, 2].plot(time, sin_theta_k[:, k])
        ax[0, 2].set_xlabel("t")
        ax[0, 2].set_ylabel(r"$\sin(\theta_k)$")
        
        ## Plot F_k and P_k
        counts, bin_edges = np.histogram(Phi_k_dot[:, k], bins = 500)
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
        bin_width = bin_edges[1] - bin_edges[0]
        pdf = counts / (np.sum(counts) * bin_width)
        ax[1, 1].plot(bin_centres, pdf)
        ax[1, 1].set_xlabel(r"$\dot{\Phi_k}$")
        ax[1, 1].set_ylabel(r"PDF")
        ax[1, 1].set_yscale('log')


        ## Plot \dot(\Phi_k) over time
        ax[1, 2].plot(time, Phi_k_dot[:, k])
        ax[1, 2].set_xlabel(r"$t$")
        ax[1, 2].set_ylabel(r"$\dot{\Phi_k}$")
        
        plt.savefig(output_dir + "/AdlerData_k[{}].png".format(k), bbox_inches='tight')
        plt.close()



    ################################
    ##  Plot Window Averaged Data
    ################################
    ## Set window sizes
    steps = [5, 10, 25, 50]

    ## Allocate arrays
    num_t       = scale_order_param.shape[0]

    ## Loop over data and average over window
    for k in k_indx:

        ## Get adler data
        theta_k       = np.mod(np.angle(scale_order_param[:, k]) + 2.0 * np.pi, 2.0 * np.pi)
        P_k           = np.absolute(scale_order_param[:, k]) 
        F_k           = k / (2.0 * a_k[k]) * P_k[:]
        sin_theta_k_k = sin_theta_k[:, k]
        Phi_k_dot_k   = Phi_k_dot[:, k]
        for dt in steps:
            print("k = {} dt = {}".format(k, dt))
            t               = []
            theta_k_avg     = []
            P_k_avg         = []
            F_k_avg         = []
            sin_theta_k_avg = []
            Phi_k_dot_avg   = []
            i = 0
            while i < num_t - dt + 1:
                ## Get averaged data
                t.append(np.mean(time[i:i + dt]))
                theta_k_avg.append(np.mean(theta_k[i:i + dt]))
                P_k_avg.append(np.mean(P_k[i:i + dt]))
                F_k_avg.append(np.mean(F_k[i:i + dt]))
                sin_theta_k_avg.append(np.mean(sin_theta_k_k[i:i + dt]))
                Phi_k_dot_avg.append(np.mean(Phi_k_dot_k[i:i + dt]))
                i += dt

            ## Plot average data
            fig, ax = plt.subplots(ncols=3, nrows=2, figsize = (16, 9), gridspec_kw = dict(hspace = 0.2, wspace = 0.3))

            ## Plot t series
            ax[0, 0].plot(t, theta_k_avg)
            ax[0, 0].set_xlabel("t")
            ax[0, 0].set_ylabel(r"$\theta_k$")

            ## Plot histogram
            counts, bin_edges = np.histogram(theta_k_avg, bins = 500)
            bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            bin_width = bin_edges[1] - bin_edges[0]
            pdf = counts / (np.sum(counts) * bin_width)
            ax[0, 1].plot(bin_centres, pdf, xunits = radians)
            ax[0, 1].set_xlabel(r"$\theta_k$")
            ax[0, 1].set_ylabel(r"PDF")
            ax[0, 1].set_yscale('log')

            ## Plot histogram of P_k and F_k and P_k over time
            div3   = make_axes_locatable(ax[1, 0])
            axtop3 = div3.append_axes("top", size = "100%", pad = 0.4)
            counts, bin_edges = np.histogram(P_k_avg, bins = 500)
            bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            bin_width = bin_edges[1] - bin_edges[0]
            pdf = counts / (np.sum(counts) * bin_width)
            axtop3.plot(bin_centres, pdf)
            axtop3.set_xlabel(r"P_k")
            axtop3.set_yscale('log')
            ax[1, 0].plot(t, P_k_avg, yunits = radians)
            ax[1, 0].plot(t, F_k_avg, yunits = radians)
            ax[1, 0].set_xlabel("t")
            ax[1, 0].set_ylabel(r"Amplitude Data")
            ax[1, 0].legend([r"$P_k$", r"$F_k$"])
            ax[1, 0].set_yscale('log')

            ## Plot sin(theta_k)
            ax[0, 2].plot(t, sin_theta_k_avg)
            ax[0, 2].set_xlabel("t")
            ax[0, 2].set_ylabel(r"$\sin(\theta_k)$")
            
            ## Plot F_k and P_k
            counts, bin_edges = np.histogram(Phi_k_dot_avg, bins = 500)
            bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            bin_width = bin_edges[1] - bin_edges[0]
            pdf = counts / (np.sum(counts) * bin_width)
            ax[1, 1].plot(bin_centres, pdf)
            ax[1, 1].set_xlabel(r"$\dot{\Phi_k}$")
            ax[1, 1].set_ylabel(r"PDF")
            ax[1, 1].set_yscale('log')

            ## Plot \dot(\Phi_k) over t
            ax[1, 2].plot(t, Phi_k_dot_avg)
            ax[1, 2].set_xlabel(r"$t$")
            ax[1, 2].set_ylabel(r"$\dot{\Phi_k}$")
            
            plt.grid(True)
            plt.suptitle(r"Window Size: {}$\Delta t$".format(dt))
            plt.savefig(output_dir + "/AdlerData_WindowAveraged_k[{}]_window[{}].png".format(k, dt), bbox_inches='tight')
            plt.close()

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
# mpl.rcParams['figure.figsize']    = [5, 3]
# # mpl.rcParams['figure.autolayout'] = True
# mpl.rcParams['text.usetex']       = True
# mpl.rcParams['font.family']       = 'serif'
# mpl.rcParams['font.serif']        = 'Computer Modern Roman'
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
from scipy.stats import linregress
import itertools 
from functions import open_file

from numpy.polynomial.polynomial import polyfit



def best_fit(X, Y):

    xbar = np.mean(X) # sum(X)/len(X)
    ybar =np.mean(Y) #sum(Y)/len(Y)
    
    n = len(X) # or len(Y)

    numer = np.sum([xi*yi for xi,yi in zip(X, Y)]) - n * (xbar * ybar) 
    denum = np.sum([xi**2 for xi in X]) - n * (xbar**2)

    b = numer / denum
    a = ybar - b * xbar

    # print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

def open_file_lyap(N, k0, a, beta, u0, iters, m_end, m_iter, trans, numLEs, func_type):
    
    HDFfileData = -1
    dof = int(N /2 - k0)
    
    ## Check if file exists and open
    if numLEs == 1:
        ## Create filename from data
        filename = input_dir + "/PAPER_LCEData_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{},{},{}]".format(N, k0, a, beta, u0, iters, m_end, m_iter)

        ## Check if file exists and open
        if os.path.exists(filename + "_TRANS[{}]_LEs[{}].h5".format(trans, numLEs)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]_LEs[{}].h5".format(trans, numLEs), 'r')
        elif os.path.exists(filename + "_TRANS[{}]_LEs[{}].h5".format(trans * 10, numLEs)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]_LEs[{}].h5".format(trans * 10, numLEs), 'r')
        elif os.path.exists(filename + "_TRANS[{}]_LEs[{}].h5".format(trans * 100, numLEs)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]_LEs[{}].h5".format(trans * 100, numLEs), 'r')
        elif os.path.exists(filename + "_TRANS[{}]_LEs[{}].h5".format(trans / 100, numLEs)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]_LEs[{}].h5".format(trans, numLEs), 'r')
        else: 
            print("File doesn't exist, check parameters!")
#             sys.exit()        
    else:
        ## Create filename from data
        filename = input_dir + "/PAPER_LCEData_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{},{},{}]".format(N, k0, a, beta, u0, iters, m_end, m_iter)

        if os.path.exists(filename + "_TRANS[{}].h5".format(trans)):
            HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans), 'r')
        elif os.path.exists(filename + "_TRANS[{}].h5".format(trans * 10)):
            HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans * 10), 'r')
        elif os.path.exists(filename + "_TRANS[{}].h5".format(trans / 10)):
            HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans / 10), 'r')
        elif os.path.exists(filename + "_TRANS[{}].h5".format(trans * 100)):
            HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans * 100), 'r')
        elif os.path.exists(filename + "_TRANS[{}]".format(trans) + "_LEs[{}].h5".format(dof)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]".format(trans) + "_LEs[{}].h5".format(dof), 'r')
        elif os.path.exists(filename + "_TRANS[{}]".format(trans * 10) + "_LEs[{}].h5".format(dof)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]".format(trans * 10) + "_LEs[{}].h5".format(dof), 'r')
        elif os.path.exists(filename + "_TRANS[{}]".format(trans / 10) + "_LEs[{}].h5".format(dof)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]".format(trans / 10) + "_LEs[{}].h5".format(dof), 'r')
        elif os.path.exists(filename + "_TRANS[{}]".format(trans * 100) + "_LEs[{}].h5".format(dof)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]".format(trans * 100) + "_LEs[{}].h5".format(dof), 'r')
        elif os.path.exists(filename + "_TRANS[{}]".format(trans * 1000) + "_LEs[{}].h5".format(dof)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]".format(trans * 1000) + "_LEs[{}].h5".format(dof), 'r')
        else: 
            print("File doesn't exist, check parameters!")
#             sys.exit()

    return HDFfileData




@njit
def compute_clv_stats_data(clv, a_k, num_tsteps, kmin, dof, numLEs):
    
    ## Memory Allocation
    v_k      = np.zeros((dof, dof))
    p_k      = np.zeros((dof, dof))
    v_k_proj = np.zeros((dof, dof))

    ## Translation Invariant Direction -> T
    T              = np.arange(2.0, float(dof + 2), 1.0)
    T_a_k          = T * a_k[kmin:]
    T_norm_sqr     = np.linalg.norm(T) ** 2
    T_enorm_sqr = np.linalg.norm(T_a_k) ** 2
    
    ## Loop over time
    for t in range(num_tsteps):

        ## Loop over vectors
        for j in range(numLEs):
            
            ## Square each component
            v_k[:, j] += np.square(clv[t, :, j])

            ## Compute the projection
            v_proj  = clv[t, :, j] - (T * (np.dot(clv[t, :, j], T))) / T_norm_sqr
            clv_a_k = clv[t, :, j] * a_k[kmin:]
            v_enorm = clv_a_k - (T_a_k * np.dot(clv_a_k, T_a_k)) / T_enorm_sqr
            
            ## Renormalize after projection
            v_proj     = v_proj / np.linalg.norm(v_proj)
            v_enorm = v_enorm / np.linalg.norm(v_enorm)
            
            ## Update running sum
            p_k[:, j]      += np.square(v_enorm)
            v_k_proj[:, j] += np.square(v_proj)
            
    ## Compute averages
    v_k       = v_k / num_tsteps
    p_k       = p_k / num_tsteps
    v_k_proj  = v_k_proj / num_tsteps
    

    return v_k, v_k_proj, p_k


if __name__ == '__main__':
    #########################
    ##  Get Input Parameters
    #########################
    k0    = 1
    N     = [2048, 4096, 8192, 16384, 2 * 16384]
    alpha = [0.0, 0.5, 1.0, 1.2, 1.5, 1.7, 2.0, 2.2, 2.5]
    beta  = 0.0
    u0    = 'RANDOM'
    iters = int(1e7)
    trans = 100000000
    
    ######################
    ##  Input & Output Dir
    ######################
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    # output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/PhaseOrder"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/Paper"

    Polynomial = np.polynomial.Polynomial



    from matplotlib import rc, rcParams
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Sans']})
    rc('text', usetex=True)
    rcParams['lines.linewidth'] = 0.75
    rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                       r'\usepackage{xcolor}',    #for \text command
                                       r'\usepackage{helvet}',    # set the normal font here
                                       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
                                       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
                                       ] 
    labelsize = 8
    ticksize  = 6







    # --------------------------------
    # Plot R_k 
    # --------------------------------
    fig = plt.figure(figsize = (2. * (3 + 3/8), 3.5), tight_layout = False)
    xg = 3
    yg = 3
    gs  = GridSpec(xg, yg, hspace = 0.3, wspace = 0.1)  

    ax = []
    for i in range(xg):
        for j in range(yg):
            ax.append(fig.add_subplot(gs[i, j]))

    lnN = np.zeros((len(alpha), len(N)))
    
    for i, a in enumerate(alpha):
        for j, n in enumerate(N):   

            file = open_file(a, n, k0, beta, u0, iters, trans, input_dir)

            if 'R_k_avg' not in list(file.keys()):
                print("No Data N = {}, a = {}".format(n, a))
                continue
            else:
                R_k_avg = file['R_k_avg']

            ax[i].plot(range(k0 + 1, int(n / 2) + 1), R_k_avg[k0 + 1:], linewidth = 0.75)
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')

            if i < 6:
                ax[i].set_xticks([])
                ax[i].set_xticklabels([])
            if np.mod(i, 3) != 0:
                ax[i].set_yticks([])
                ax[i].set_yticklabels([])
        

         
            lnN[i, j] = np.sum(R_k_avg[k0 + 1:int(N[0] /2)])


        for i in range(3):
            ax[(yg - 1) * 3 + i].set_xlabel(r"$k$", fontsize = labelsize)


        for i in range(3):
            ax[i * 3 + 0].set_ylabel(r"$R_k$", fontsize = labelsize)
            # ax[i * 3 + 0].set_yticks([0, 0.5, 1])
            # ax[i * 3 + 0].set_yticklabels([0, 0.5, 1.0])

    for i, a in enumerate(alpha):
        if i <= 2:
            plt.gcf().text(0.15  + 0.3 * np.mod(i, 3), 0.85, r"$\alpha = {:0.2f}$".format(a), fontsize = labelsize)
        elif i > 2 and i < 6:
            plt.gcf().text(0.2  + 0.3 * np.mod(i, 3), 0.55, r"$\alpha = {:0.2f}$".format(a), fontsize = labelsize)
        else: 
            plt.gcf().text(0.2  + 0.3 * np.mod(i, 3), 0.3, r"$\alpha = {:0.2f}$".format(a), fontsize = labelsize)
        # ax[i].set_ylim(0, 1)
        ax[i].set_xlim(k0 + 1, int(n / 2))
        ax[i].tick_params(axis='x', which = 'both', bottom = True, labelsize = ticksize)
        ax[i].tick_params(axis='y', which = 'both', bottom = True, labelsize = ticksize)
        


    # for i in range(xg * yg):
    #     ax[i].set_xlim(k0 + 1, 100)

    ax[0].legend([r"$N = {}$".format(int(n / 2)) for n in N], fontsize = ticksize, bbox_to_anchor = (0.6, 1.0, 1, 0.2), loc="lower left", ncol = len(N))
    plt.savefig(output_dir + "/R_k_avg_LogLog_N2_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()
    




    N     = [128, 256, 512, 1024]
    k0    = 1
    beta  = 0.0
    iters = 400000
    m_end = 8000
    m_itr = 50
    trans = 0
    u0    = "RANDOM"

    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/PRL_PAPER_DATA/Plots"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/Paper"


    fig = plt.figure(figsize = (2. * (3 + 3/8), 3.5), tight_layout = False)
    xg = 3
    yg = 3
    gs  = GridSpec(xg, yg, hspace = 0.3, wspace = 0.1)  

    ax = []
    for i in range(xg):
        for j in range(yg):
            ax.append(fig.add_subplot(gs[i, j]))



    ## --------- Loop over Data
    for i, n in enumerate(N):

        krange = range(k0 + 1, int(n / 2 + 1))

        for j, a in enumerate(alpha):

            print("N = {}, a = {:0.2f}".format(n, a))

            ## Open data file
            HDFfileData = open_file_lyap(n, k0, a, beta, u0, iters, m_end, m_itr, 1000, int(n / 2 - k0), "max")
            if HDFfileData == -1:
                continue


            ## ------- Read in Parameter Data
            amps    = HDFfileData['Amps'][:]
            kmin    = k0 + 1
            num_osc = amps.shape[0]
            dof     = num_osc - kmin

            ## ----------------- ##
            ## ------ CLVs ----- ##
            ## ----------------- ##
            ## Read in CLVs
            CLVs          = HDFfileData['CLVs']
            clv_dims      = CLVs.attrs['CLV_Dims']
            num_clv_steps = CLVs.shape[0]            
            clv           = np.reshape(CLVs, (CLVs.shape[0], dof, dof))

            ## Compute projected vectors
            v_k, v_k_proj, p_k = compute_clv_stats_data(clv, amps, num_clv_steps, kmin, dof, int(n / 2 - k0))


            ax[j].plot(krange, v_k[:, 0])
            ax[j].set_yscale('log')
            ax[j].set_xscale('log')

            if j < 6:
                ax[j].set_xticks([])
                ax[j].set_xticklabels([])
            if np.mod(j, 3) != 0:
                ax[j].set_yticks([])
                ax[j].set_yticklabels([])

            for i in range(3):
                ax[(yg - 1) * 3 + i].set_xlabel(r"$k$", fontsize = labelsize)

            for i in range(3):
                ax[i * 3 + 0].set_ylabel(r"$\langle v_k^2 \rangle$", fontsize = labelsize)
                # ax[i * 3 + 0].set_yticks([0, 0.5, 1])
                # ax[i * 3 + 0].set_yticklabels([0, 0.5, 1.0])

    for i, a in enumerate(alpha):
        if i <= 2:
            plt.gcf().text(0.13  + 0.3 * np.mod(i, 3), 0.85, r"$\alpha = {:0.2f}$".format(a), fontsize = labelsize)
        elif i > 2 and i < 6:
            plt.gcf().text(0.15  + 0.3 * np.mod(i, 3), 0.4, r"$\alpha = {:0.2f}$".format(a), fontsize = labelsize)
        else: 
            plt.gcf().text(0.15  + 0.3 * np.mod(i, 3), 0.125, r"$\alpha = {:0.2f}$".format(a), fontsize = labelsize)
        ax[i].tick_params(axis='x', which = 'both', bottom = True, labelsize = ticksize)
        ax[i].tick_params(axis='y', which = 'both', bottom = True, labelsize = ticksize)

    ax[0].legend([r"$N = {}$".format(int(n / 2)) for n in N], fontsize = ticksize, bbox_to_anchor = (0.8, 1.0, 1, 0.2), loc="lower left", ncol = len(N))
    plt.savefig(output_dir + "/CLV_SM_LogLog_N2_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()


    # ## --------------------------------
    # ## Plot scaling of R_k
    # ## --------------------------------
    # fig = plt.figure(figsize = (7, 2.4), tight_layout = False)
    # xg = 3object, *
    # yg = 3
    # gs  = GridSpec(xg, yg)  


    # Polynomial = np.polynomial.Polynomial

    # ax = []
    # for i in range(xg):
    #     for j in range(yg):
    #         ax.append(fig.add_subplot(gs[i, j]))

    # nn = [int(N[i] / 2 - k0) for i in range(len(N))]
    # for i, a in enumerate(alpha):
    #     intercept, slope_bf = best_fit(np.log(lnN[i, :]), np.log(nn))
    #     slope_rr = (np.log(nn[-1]) - np.log(nn[0])) / (np.log(lnN[i, -1]) - np.log(lnN[i, 0]))
    #     pfit, stats = Polynomial.fit(np.log(lnN[i, :]), np.log(nn), 1, full = True)#, window = (min(nn), max(nn)), domain = (min(nn), max(nn)))
    #     intercept, slope_poly = pfit
    #     b, m = polyfit(np.log(lnN[i, :]), np.log(nn), 1)
    #     print("a = {} || slope_bf = {}, slope_rr = {}, slope_poly = {}, slope_polyfit = {}".format(a, slope_bf, slope_rr, slope_poly, m))
        
    #     for j in range(len(N)):
    #         ax[i].plot(np.log(lnN[i, j]), np.log(nn[j]), '.')
    #     # ax[i].plot([intercept + slope * np.log(n) for n in N], np.log(nn),  '--', lw = 0.5, color = 'black')
    #     # ax[i].plot((np.mean(np.log(nn)) - slope * np.mean(np.log(lnN))) + np.log(nn ** slope), np.log(nn),  '--', lw = 0.5, color = 'black')
    #     ax[i].text(np.log(lnN[i, 0]), np.log(nn[0]), r"$\alpha = {:0.2f}$".format(a))
    #     if i <= 5:
    #         ax[i].set_xticks([])
    #         ax[i].set_xticklabels([])
    #     if np.mod(i, 3) != 0:
    #         ax[i].set_yticks([])
    #         ax[i].set_yticklabels([])

    # for i in range(3):
    #     ax[(yg - 1) * 3 + i].set_xlabel(r"$ln(\sum R_k)$")

    # for i in range(3):
    #     ax[i * 3 + 0].set_ylabel(r"$ln(N)$")

    # ax[0].legend([r"$N = {}$".format(n) for n in N], bbox_to_anchor = (0.05, 1.0, 1, 0.2), loc="lower left", ncol = len(N))
    # plt.savefig(output_dir + "/R_k_SCALING_N2_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()




    # nn = [int(N[i] / 2 - k0) for i in range(len(N))]
 

    # ## --------------------------------
    # ## Plot B(alpha)
    # ## --------------------------------
    # alpha = np.arange(0.0, 2.51, 0.1)
    # lnN = np.zeros((len(N), ))
    
    # slope_rr   = np.zeros((len(alpha), ))
    # slope_bf   = np.zeros((len(alpha), ))
    # slope_poly = np.zeros((len(alpha), ))

    # for i, a in enumerate(alpha):
    #     for j, n in enumerate(N):

    #         file = open_file(a, n, k0, beta, u0, iters, trans, input_dir)

    #         R_k_avg = file['R_k_avg']

    #         lnN[j] = np.sum(R_k_avg[k0 + 1:int(N[0] /2)])

    #     slope_rr[i] =  (np.log(nn[-1]) - np.log(nn[0])) / (np.log(lnN[-1]) - np.log(lnN[0])) # (lnN[-1] - lnN[0]) / (nn[-1] - nn[0])
        
    #     intercept, slope_bf[i] = best_fit(np.log(lnN[:]), np.log(nn))
        
    #     pfit, stats = Polynomial.fit(np.log(lnN[:]), np.log(nn), 1, full = True)#, window = (min(nn), max(nn)), domain = (min(nn), max(nn)))
    #     intercept, slope_poly[i] = pfit
    
    # fig = plt.figure(figsize = (7, 2.4), tight_layout = False)
    # plt.plot(alpha, slope_rr)
    # plt.plot(alpha, slope_bf)
    # plt.plot(alpha, slope_poly)
    # plt.ylim(0.0, 2.5)
    # plt.xlabel(r"$\alpha$")
    # plt.ylabel(r"$B(\alpha)$")
    # plt.legend([r"Rise / Run", r"Least Sq", "Poly"])

    # plt.savefig(output_dir + "/R_k_SCALING_N2_FULL_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()











    # ## N
    # N = [1024, 2048, 4096, 8192, 16384, 32768]

    # ## Window length
    # win = 3

    # ## Data
    # bf_slopes   = np.zeros((len(alpha), len(N) - 2))
    # poly_slopes = np.zeros((len(alpha), len(N) - 2))

    # ## Create figure
    # fig = plt.figure(figsize = (16, 9), tight_layout = False)
    # xg = 3
    # yg = 3
    # gs  = GridSpec(xg, yg, hspace = 0.3, wspace = 0.1)  

    # ax = []
    # for i in range(xg):
    #     for j in range(yg):
    #         ax.append(fig.add_subplot(gs[i, j]))

    # for i, a in enumerate(alpha):
    #     for p in range(1, len(N) - 1):
            
    #         ## Open files
    #         file  = []
    #         for o in [-1, 0, 1]:
    #             file.append(open_file(a, N[p + o], k0, beta, u0, iters, trans, input_dir))
            
    #         ## Read in data and compute log sum
    #         lnSum = []
    #         for o in range(win):
    #             lnSum.append(np.log(np.sum(file[o]['R_k_avg'][:])))

    #         ## Compute the slopes using two methods     
    #         intercept, bf_slopes[i, p - 1 ] = best_fit(lnSum, np.log(int(N[(p - 1):(p + 1) + 1] / 2 + k0)))
            
    #         pfit, stats = Polynomial.fit(lnSum, np.log(int(N[(p - 1):(p + 1) + 1] / 2 + k0), 1, full = True)#, window = (min(nn), max(nn)), domain = (min(nn), max(nn)))
    #         intercept, poly_slopes[i, p - 1] = pfit

    #     ax[i].plot(bf_slopes[i, :], '.-')
    #     ax[i].plot(poly_slopes[i, :], 'o-')
    #     # ax[i].legend([r"Best Fit", r"Poly Fit"])

    # ax[0].legend([r"Best Fit", r"Poly Fit"], bbox_to_anchor = (0.05, 1.0, 1, 0.2), loc="lower left", ncol = len(N))
    # plt.savefig(output_dir + "/R_k_Intemediate_slope_SCALING_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()
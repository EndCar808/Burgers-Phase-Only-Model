#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros

######################
##	Library Imports
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
# mpl.rcParams['lines.linewidth']   = 1.25
# mpl.rcParams['lines.markersize']  = 6
mpl.rcParams['axes.labelsize'] = 10
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
from cycler import cycler

#########################
##    Function Defs    ##
#########################
@njit
def compute_R_alpha(R_k, a_k, num_osc, kmin, comp_type):

    ## Allocate array
    R_k_avg = np.zeros((num_osc, ))

    ## Get the time average 
    for k in range(R_k.shape[1]):
        tmp = 0.0
        for t in range(kmin, R_k.shape[0]):
            tmp += R_k[t, k]
        R_k_avg[k] = tmp / R_k.shape[0]

    ## Get the weighting array for averaging
    if comp_type == "avg":
        amp = np.ones((num_osc, ))
    elif comp_type == "w_avg":
        amp = a_k

    if comp_type == "w_avg" or comp_type == "avg":
        ## Compute avarage
        norm    = 0
        R_alpha = 0.0
        for i in range(kmin, num_osc):
            R_alpha += amp[i] * R_k_avg[i]
            norm    += 1

        R_alpha /= norm

        return R_alpha, 0
    elif comp_type == "max": 
        return np.amax(R_k_avg), np.argmax(R_k_avg)


@njit
def amp_normalization(amps, n, k0):

    ## Allocate array
    norm = np.zeros((n, ))

    ## Compute the normalization factor
    for kk in range(0, n):
        if kk <= k0:
            norm[kk] = 0.0
        else:
            for k1 in range(-(n - 1) + kk, n):
                norm[kk] += amps[np.absolute(k1)] * amps[np.absolute(kk - k1)]

    return norm



#########################
##        Main         ##
#########################
if __name__ == '__main__':
    

    ############################
    ##    Input Parameters    ##
    ############################
    k0    = 1
    alpha = np.arange(0.0, 2.51, 0.1)
    beta  = 0.0
    iters = int(4e5)
    trans = 10000000
    N     = 2048
    u0    = "RANDOM"

    a_amp = np.arange(0.0, 2.55, 0.1)
    # a_amp = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5]

    ## Input and output directories
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/PhaseOrder"

   
    # #######################
    # ##  Allocate Memory  ##
    # #######################
    # ## Computation type
    # comp_type = ["avg"] #, "w_avg", "max"]

    # ## Create dfs
    # R_a            = pd.DataFrame(columns = comp_type)
    # R_adler_a      = pd.DataFrame(columns = comp_type)
    # R_phaseshift_a = pd.DataFrame(columns = comp_type)
    # R_agustins_a   = pd.DataFrame(columns = comp_type)

    # k_R_a            = pd.DataFrame(columns = comp_type)
    # k_R_adler_a      = pd.DataFrame(columns = comp_type)
    # k_R_phaseshift_a = pd.DataFrame(columns = comp_type)
    # k_R_agustins_a   = pd.DataFrame(columns = comp_type)

    # for c in comp_type:
    #     R_a[c]            = np.zeros((alpha.shape))
    #     R_adler_a[c]      = np.zeros((alpha.shape))
    #     R_phaseshift_a[c] = np.zeros((alpha.shape))
    #     R_agustins_a[c]   = np.zeros((alpha.shape))

    #     k_R_a[c]            = np.zeros((alpha.shape))
    #     k_R_adler_a[c]      = np.zeros((alpha.shape))
    #     k_R_phaseshift_a[c] = np.zeros((alpha.shape))
    #     k_R_agustins_a[c]   = np.zeros((alpha.shape))

    # R_a_split            = np.zeros((3, alpha.shape[0]))
    # R_adler_a_split      = np.zeros((3, alpha.shape[0]))
    # R_phaseshift_a_split = np.zeros((3, alpha.shape[0]))
    # R_agustins_a_split   = np.zeros((3, alpha.shape[0]))

    # amps           = np.zeros((int(N/2 + 1), len(a_amp)))
    # coupling_const = np.zeros((int(N/2 + 1), len(a_amp)))
    # a_indx = 0

    # amp_norm_k      = np.zeros((len(a_amp), int(N/2 + 1)))
    # amp_norm_k_full = np.zeros((alpha.shape[0], int(N/2 + 1)))

    # for i, a in enumerate(alpha):

    #     ## Get Filename
    #     filename   = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(N, k0, a, beta, u0, iters, trans)
    #     filename10 = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(N, k0, a, beta, u0, iters, int(trans/10))

    #     ## Open in current file
    #     if os.path.exists(input_dir + filename + '.h5'):
    #         HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')
    #     elif os.path.exists(input_dir + filename10 + '.h5'):
    #         HDFfileData = h5py.File(input_dir + filename10 + '.h5', 'r')
    #     else:
    #         print("File doesn't exist!...Alpha = {:.3f}".format(a))
        
    #     ## Read in data
    #     # Amplitudes
    #     a_k     = HDFfileData['Amps'][:]
    #     num_osc = a_k.shape[0]
    #     kmin    = k0 + 1
    #     kmax    = num_osc - 1

    #     amp_norm_k_full[i, :] = amp_normalization(a_k, num_osc, k0)

    #     if np.any(a_amp == a):
    #         amps[:, a_indx] = a_k

    #         amp_norm = amp_normalization(a_k, num_osc, k0)
    #         for k in range(int(N/2 + 1)):
    #             if k <= k0:
    #                 coupling_const[k, a_indx] = 0.0
    #             else:
    #                 coupling_const[k, a_indx] = (-k / a_k[k]) *  amp_norm[k]

    #         # Compute the normaliztion factor
    #         amp_norm_k[a_indx, :] = amp_normalization(a_k, num_osc, k0)

    #         a_indx += 1

    #     ## Scale dependent phase order parameters
    #     order_k            = HDFfileData['ScaleOrderParam'][:, :] 
    #     adler_order_k      = HDFfileData['AdlerScaleOrderParam'][:, :]
    #     PhaseShift_order_k = HDFfileData['PhaseShiftScaleOrderParam'][:, :]
    #     agustins_order_k   = HDFfileData['AgustinsScaleOrderParam'][:, :]

    #     ## Extract the sync parameter R_k
    #     R_k            = np.absolute(order_k)
    #     R_adler_k      = np.absolute(adler_order_k) 
    #     R_phaseshift_k = np.absolute(PhaseShift_order_k) 
    #     R_agustins_k   = np.absolute(agustins_order_k) 

    #     ## Compute the R(alpha) parameter
    #     for c in comp_type:
    #         R_a[c][i], k_R_a[c][i]                       = compute_R_alpha(R_k, a_k, num_osc, k0, c)
    #         R_adler_a[c][i], k_R_adler_a[c][i]           = compute_R_alpha(R_adler_k, a_k, num_osc, k0, c)
    #         R_phaseshift_a[c][i], k_R_phaseshift_a[c][i] = compute_R_alpha(R_phaseshift_k, a_k, num_osc, k0, c)
    #         R_agustins_a[c][i], k_R_agustins_a[c][i]     = compute_R_alpha(R_agustins_k, a_k, num_osc, k0, c)

    #     # Compute the R(alpha) parameter for low and high wavenumbers
    #     R_a_split[0, i], q            = compute_R_alpha(R_k[:, :int(num_osc/4)], a_k[:int(num_osc/4)], num_osc, k0, "avg")
    #     R_adler_a_split[0, i], q      = compute_R_alpha(R_adler_k[:, :int(num_osc/4)], a_k[:int(num_osc/4)], num_osc, k0, "avg")
    #     R_phaseshift_a_split[0, i], q = compute_R_alpha(R_phaseshift_k[:, :int(num_osc/4)], a_k[:int(num_osc/4)], num_osc, k0, "avg")
    #     R_agustins_a_split[0, i], q   = compute_R_alpha(R_agustins_k[:, :int(num_osc/4)], a_k[:int(num_osc/4)], num_osc, k0, "avg")

    #     R_a_split[1, i], q            = compute_R_alpha(R_k[:, int(num_osc/4):int(3*num_osc/4)], a_k[int(num_osc/4):int(3*num_osc/4)], num_osc, k0, "avg")
    #     R_adler_a_split[1, i], q      = compute_R_alpha(R_adler_k[:, int(num_osc/4):int(3*num_osc/4)], a_k[int(num_osc/4):int(3*num_osc/4)], num_osc, k0, "avg")
    #     R_phaseshift_a_split[1, i], q = compute_R_alpha(R_phaseshift_k[:, int(num_osc/4):int(3*num_osc/4)], a_k[int(num_osc/4):int(3*num_osc/4)], num_osc, k0, "avg")
    #     R_agustins_a_split[1, i], q   = compute_R_alpha(R_agustins_k[:, int(num_osc/4):int(3*num_osc/4)], a_k[int(num_osc/4):int(3*num_osc/4)], num_osc, k0, "avg")

    #     R_a_split[2, i], q            = compute_R_alpha(R_k[:, int(3*num_osc/4):], a_k[int(3*num_osc/4):], num_osc, k0, "avg")
    #     R_adler_a_split[2, i], q      = compute_R_alpha(R_adler_k[:, int(3*num_osc/4):], a_k[int(3*num_osc/4):], num_osc, k0, "avg")
    #     R_phaseshift_a_split[2, i], q = compute_R_alpha(R_phaseshift_k[:, int(3*num_osc/4):], a_k[int(3*num_osc/4):], num_osc, k0, "avg")
    #     R_agustins_a_split[2, i], q   = compute_R_alpha(R_agustins_k[:, int(3*num_osc/4):], a_k[int(3*num_osc/4):], num_osc, k0, "avg")




    # #######################
    # ##     Plot Data     ##
    # #######################

    # #########
    # ## All ##
    # #########
    # fig = plt.figure(figsize = (28, 7.5), tight_layout = False)
    # gs  = GridSpec(3, 4)

    # ax = []
    # for i, c in enumerate(comp_type):
    #     for j in range(4):
    #         ax.append(fig.add_subplot(gs[i, j]))
     

    # for i, c in enumerate(comp_type):
    #     if i == 0:
    #         ax[0].set_title(r"$-i\frac{\sum_{k_1} a_{k_1} a_{k - k_1}e^{i(\phi_{k_1} + \phi_{k - k_1})}}{\sum_{k_1}a_{k_1}a_{k - k_1}}$")
    #         ax[1].set_title(r"$-i(sgn k)e^{i\phi_k}\frac{\sum_{k_1}a_{k_1}a_{k - k_1}e^{-i\varphi_{k_1, k - k_1}}}{\sum_{k_1}a_{k_1}a_{k - k_1}}$")
    #         ax[2].set_title(r"$\frac{\sum_{k_1}\frac{-k a_{k_1}a_{k - k_1}}{a_k}e^{i\varphi_{k_1, k - k_1}}}{(-k / a_k)\sum_{k_1}a_{k_1}a_{k - k_1}}$")
    #         ax[3].set_title(r"$i(sgn k)\frac{\sum_{k_1}a_{k_1}a_{k - k_1}e^{i\varphi_{k_1, k - k_1}}}{\sum_{k_1}a_{k_1}a_{k - k_1}}$")

    #     ax[i * 4 + 0].plot(alpha, R_a[c], '-')
    #     ax[i * 4 + 0].legend([r"Order"])
    #     ax[i * 4 + 0].set_xlabel(r"$\alpha$")
    #     ax[i * 4 + 0].set_ylabel(r"$\mathcal{R}(\alpha)$")
        
    #     ax[i * 4 + 1].plot(alpha, R_adler_a[c], '-')
    #     ax[i * 4 + 1].legend([r"Adler"])
    #     ax[i * 4 + 1].set_xlabel(r"$\alpha$") 
    #     ax[i * 4 + 1].set_ylabel(r"$\mathcal{R}(\alpha)$")
        
    #     ax[i * 4 + 2].plot(alpha, R_agustins_a[c], '-')
    #     ax[i * 4 + 2].legend([r"Agustins"])
    #     ax[i * 4 + 2].set_xlabel(r"$\alpha$")
    #     ax[i * 4 + 2].set_ylabel(r"$\mathcal{R}(\alpha)$")
        
    #     ax[i * 4 + 3].plot(alpha, R_phaseshift_a[c], '-')
    #     ax[i * 4 + 3].legend([r"Phase Shift"])
    #     ax[i * 4 + 3].set_xlabel(r"$\alpha$")
    #     ax[i * 4 + 3].set_ylabel(r"$\mathcal{R}(\alpha)$")



    # plt.savefig(output_dir + "/PHASE_ORDER_PARAMETERS_ALL_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()

    # ############################
    # ## Split wavenumber space ##
    # ############################
    # fig = plt.figure(figsize = (24, 7.5), tight_layout = False)
    # gs  = GridSpec(1, 4)

    # ax = []
    # for j in range(4):
    #     ax.append(fig.add_subplot(gs[i, j]))

    # ax[0].set_title(r"$-i\frac{\sum_{k_1} a_{k_1} a_{k - k_1}e^{i(\phi_{k_1} + \phi_{k - k_1})}}{\sum_{k_1}a_{k_1}a_{k - k_1}}$")
    # ax[1].set_title(r"$-i(sgn k)e^{i\phi_k}\frac{\sum_{k_1}a_{k_1}a_{k - k_1}e^{-i\varphi_{k_1, k - k_1}}}{\sum_{k_1}a_{k_1}a_{k - k_1}}$")
    # ax[2].set_title(r"$\frac{\sum_{k_1}\frac{-k a_{k_1}a_{k - k_1}}{a_k}e^{i\varphi_{k_1, k - k_1}}}{(-k / a_k)\sum_{k_1}a_{k_1}a_{k - k_1}}$")
    # ax[3].set_title(r"$i(sgn k)\frac{\sum_{k_1}a_{k_1}a_{k - k_1}e^{i\varphi_{k_1, k - k_1}}}{\sum_{k_1}a_{k_1}a_{k - k_1}}$")

    # for s in range(3):
    #     ax[0].plot(alpha, R_a_split[s, :], '-')
    # ax[0].legend([r"Order"])
    # ax[0].set_xlabel(r"$\alpha$")
    # ax[0].set_ylabel(r"$\mathcal{R}(\alpha)$")
    # ax[0].legend([r"$k < N /4$", r"$N /4 \leq k < 3N /4$", r"$3N /4 \leq k \leq N$"])
    
    # for s in range(3):
    #     ax[1].plot(alpha, R_adler_a_split[s, :], '-')
    # ax[1].legend([r"Adler"])
    # ax[1].set_xlabel(r"$\alpha$") 
    # ax[1].set_ylabel(r"$\mathcal{R}(\alpha)$")
    # ax[1].legend([r"$k < N /4$", r"$N /4 \leq k < 3N /4$", r"$3N /4 \leq k \leq N$"])
    
    # for s in range(3):
    #     ax[2].plot(alpha, R_agustins_a_split[s, :], '-')
    # ax[2].legend([r"Agustins"])
    # ax[2].set_xlabel(r"$\alpha$")
    # ax[2].set_ylabel(r"$\mathcal{R}(\alpha)$")
    # ax[2].legend([r"$k < N /4$", r"$N /4 \leq k < 3N /4$", r"$3N /4 \leq k \leq N$"])
    
    # for s in range(3):
    #     ax[3].plot(alpha, R_phaseshift_a_split[s, :], '-')
    # ax[3].legend([r"Phase Shift"])
    # ax[3].set_xlabel(r"$\alpha$")
    # ax[3].set_ylabel(r"$\mathcal{R}(\alpha)$")
    # ax[3].legend([r"$k < N /4$", r"$N /4 \leq k < 3N /4$", r"$3N /4 \leq k \leq N$"])


    # plt.savefig(output_dir + "/PHASE_ORDER_PARAMETERS_SPLIT_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()


    
    
    

    # fig = plt.figure(figsize = (32, 9), tight_layout = False)
    # gs  = GridSpec(1, 2)

    # ax1 = fig.add_subplot(gs[0, 0])
    # for i in range(len(a_amp)):
    #     ax1.plot(amp_norm_k[i, k0 + 1:], '-')
    # ax1.legend([r"$\alpha = {:.2f}$".format(a) for a in a_amp])
    # ax1.set_title(r"Amplitude Normalization Factor")
    # ax1.set_xlabel(r"$k$")
    # ax1.set_ylabel(r"$\sum_{k_1}a_{k_1, k - k_1}$")
    # ax1.set_yscale('log')

    # ax1 = fig.add_subplot(gs[0, 1])
    # for i in range(len(a_amp)):
    #     ax1.plot(amp_norm_k[i, k0 + 1:], '-')
    # ax1.legend([r"$\alpha = {:.2f}$".format(a) for a in a_amp])
    # ax1.set_title(r"Amplitude Normalization Factor")
    # ax1.set_xlabel(r"$k$")
    # ax1.set_ylabel(r"$\sum_{k_1}a_{k_1, k - k_1}$")
    # ax1.set_yscale('log')
    # ax1.set_xscale('log')

    # plt.savefig(output_dir + "/AMP_NORM_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close() 
    

    # a_norm_alpha  = np.zeros(alpha.shape)
    # a_wnorm_alpha = np.zeros(alpha.shape)
    # count         = 0

    # for i in range(len(alpha)):
    #     count = 0
    #     for k in range(k0 + 1, num_osc):
    #         a_norm_alpha[i]  += amp_norm_k_full[i, k]
    #         a_wnorm_alpha[i] += k * amp_norm_k_full[i, k]
    #         count += 1
    #     a_norm_alpha[i] /= count
    #     a_wnorm_alpha[i] /= count

    # fig = plt.figure(figsize = (16, 9), tight_layout = False)
    # gs  = GridSpec(1, 1)

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.plot(alpha, a_norm_alpha)
    # # ax1.plot(alpha, a_wnorm_alpha)
    # ax1.set_title(r"Averaged Amplitude Normalization Factor")
    # ax1.set_xlabel(r"$\alpha$")
    # # ax1.set_ylabel(r"$\alpha$")  
    # ax1.legend([r"Average", r"Weighted Average"])

    # plt.savefig(output_dir + "/AMP_NORM_FULL_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close() 
    



    # fig = plt.figure(figsize = (32, 10), tight_layout = False)
    # gs  = GridSpec(1, 3)

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.plot(alpha, k_R_a, '-')
    # ax1.legend([r"Order"])
    # ax1.set_xlabel(r"$\alpha$")
    # ax1.set_ylabel(r"$\mathcal{R}(\alpha)$")

    # ax2 = fig.add_subplot(gs[0, 1])
    # ax2.plot(alpha, k_R_adler_a, '-')
    # ax2.legend([r"Triad"])
    # ax2.set_xlabel(r"$\alpha$") 
    # ax2.set_ylabel(r"$\mathcal{R}(\alpha)$")

    # ax3 = fig.add_subplot(gs[0, 2])
    # ax3.plot(alpha, k_R_agustins_a, '-')
    # ax3.legend([r"Agustins"])
    # ax3.set_xlabel(r"$\alpha$")
    # ax3.set_ylabel(r"$\mathcal{R}(\alpha)$")
    # ax3.set_yscale('log')

    # plt.savefig(output_dir + "/PHASE_ORDER_k_MAX_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()
    


    # fig = plt.figure(tight_layout = False)
    # gs  = GridSpec(1, 1)

    # ax1 = fig.add_subplot(gs[0, 0])
    # for i in range(len(a_amp)):
    #     ax1.plot(np.arange(k0 + 1, num_osc), amps[k0+ 1:, i] * amps[k0 + 1:, i], '-')
    # ax1.set_title(r"Energy Spectrum")
    # ax1.legend([r"$\alpha = {:.2f}$".format(a) for a in a_amp])
    # ax1.set_xlabel(r"$k$")
    # ax1.set_ylabel(r"$E_k$")
    # ax1.set_yscale('log')
    # ax1.set_xscale('log')
    # ax1.set_xlim(k0 + 1, num_osc)

    # plt.savefig(output_dir + "/AMPS_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()


    # fig = plt.figure(tight_layout = False)
    # gs  = GridSpec(1, 1)

    # ax3 = fig.add_subplot(gs[0, 0])
    # for i in range(len(a_amp)):
    #     ax3.plot(np.arange(k0 + 1, num_osc), coupling_const[k0 + 1:, i], '-')
    # ax3.set_title(r"Coupling Constant")
    # ax3.legend([r"$\alpha = {:.2f}$".format(a) for a in a_amp])
    # ax3.set_xlabel(r"$k$")
    # ax3.set_ylabel(r"$\omega_{k_1, k - k_{1}}^{k}$")
    # ax3.set_yscale('symlog')
    # # ax3.set_xscale('log')
    # ax3.set_xlim(k0 + 1, num_osc)

    # plt.savefig(output_dir + "/COUPLING_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()
    






    #######################
    ##     Plot Data     ##
    # #######################
    N = [512, 1024, 2048, 4096, 8192]

    R_k_N_alpha = np.zeros((len(N), len(alpha)))
    R_k_N_alpha_cent = np.zeros((len(N), len(alpha)))
    R_k_N_alpha_half = np.zeros((len(N), len(alpha)))
    R_k_N_alpha_half_cent = np.zeros((len(N), len(alpha)))
    R_k_alpha = np.zeros(alpha.shape)

    fig = plt.figure(figsize = (16, 9), tight_layout = False)
    gs  = GridSpec(1, 2)  
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    for j, n in enumerate(N):    
        for i, a in enumerate(alpha):


            ## Get Filename
            filename     = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, trans)
            filename10   = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/10))
            filename100  = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/100))
            filename1000 = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/1000))
            filename10000 = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/10000))
            
            print("a = {:0.3f} || Filename: {}".format(a, filename))


            ## Open in current file
            if os.path.exists(input_dir + filename + '.h5'):
                HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')
            elif os.path.exists(input_dir + filename10 + '.h5'):
                HDFfileData = h5py.File(input_dir + filename10 + '.h5', 'r')
            elif os.path.exists(input_dir + filename100 + '.h5'):
                HDFfileData = h5py.File(input_dir + filename100 + '.h5', 'r')
            elif os.path.exists(input_dir + filename1000 + '.h5'):
                HDFfileData = h5py.File(input_dir + filename1000 + '.h5', 'r')
            elif os.path.exists(input_dir + filename1000 + '.h5'):
                HDFfileData = h5py.File(input_dir + filename1000 + '.h5', 'r')
            else:
                print("File doesn't exist!...Alpha = {:.3f}".format(a))
                continue
            
            ## Read in data
            # # Amplitudes
            # a_k     = HDFfileData['Amps'][:]
            # num_osc = a_k.shape[0]
            # kmin    = k0 + 1
            # kmax    = num_osc - 1

            ## Compute the amplitude normalization factor
            # amp_norm = amp_normalization(a_k, num_osc, k0)

            ## Scale dependent phase order parameter - R_k
            # Read in data
            R_k_avg = HDFfileData['R_k_avg']
            num_osc = R_k_avg.shape[0]
            # P_k_avg = HDFfileData['P_k_avg']

            if (n == 16384 or n == 32768) and (a == alpha[1] or a == alpha[2] or a == alpha[3]):
                fig = plt.figure(figsize = (16, 9), tight_layout = False)
                plt.plot(R_k_avg[k0 + 1:])
                plt.savefig(output_dir + "/R_k_N_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(n, a, beta, k0, iters, u0), bbox_inches='tight') 
                plt.close()


            # Compute the average over time and average over space
            R_k_alpha[i] = np.mean(R_k_avg) 
            R_k_N_alpha[j, i] = R_k_alpha[i]

            R_k_N_alpha_cent[j, i] = np.mean(np.arange(0, num_osc) * R_k_avg) / int(n /2 - k0)
            R_k_N_alpha_half[j, i] = np.mean(R_k_avg[:int((num_osc - 1)/2)] )
            R_k_N_alpha_half_cent[j, i] = np.mean(np.arange(0, int((num_osc - 1)/2)) * R_k_avg[:int((num_osc - 1)/2)])  / int(n /2 - k0)

            # # Plot P_k and R_k            
            # ax1.plot(np.arange(kmin, kmax + 1), amp_norm[kmin:], '-')
            # ax1.set_xlabel(r"$k$")
            # ax1.set_ylabel(r"$\sum_{k_1}a_{k_1}a_{k - k_1}$")
            # ax1.set_xscale('log')
            # ax1.set_yscale('log')
            # ax2.plot(np.arange(kmin, kmax + 1), P_k_avg[kmin:], '-')
            # ax2.set_xlabel(r"$k$")
            # ax2.set_ylabel(r"$\mathcal{P}_k$")
            # ax2.set_xscale('log')
            # ax2.set_yscale('log')

            HDFfileData.close()

        # ax1.legend([r"$\alpha =$ {:0.1f}".format(a) for a in alpha], bbox_to_anchor=(-0.1, 0.2), loc="lower right")
        # ax2.legend([r"$\alpha =$ {:0.1f}".format(a) for a in alpha], bbox_to_anchor=(1.04,1), loc="upper left")
        # plt.savefig(output_dir + "/R_k_P_k_N[{}]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(n, beta, k0, iters, u0), bbox_inches='tight') 
        # plt.close()

        # fig = plt.figure(figsize = (16, 9), tight_layout = False)
        # gs  = GridSpec(1, 1)   

        # ax1 = fig.add_subplot(gs[0, 0])
        # ax1.plot(alpha, R_k_alpha[:], '-')
        # ax1.legend([r"Order"])
        # ax1.set_xlabel(r"$\alpha$")
        # ax1.set_ylabel(r"$\mathcal{R}(\alpha)$")   

        # plt.savefig(output_dir + "/R_k_TIME_N[{}]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(n, beta, k0, iters, u0), bbox_inches='tight') 
        # plt.close()


    fig = plt.figure(figsize = (7.0, 2.7), tight_layout = False)
    gs  = GridSpec(1, 1)   
    labelsize = 10
    ticksize  = 10

    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(len(N)):
        ax1.plot(alpha, R_k_N_alpha[i, :], '-', linewidth = 1.0)
    ax1.set_xlabel(r"$\alpha$", fontsize = labelsize)
    ax1.set_ylabel(r"$\mathcal{R}$", fontsize = labelsize)   
    ax1.tick_params(labelsize = ticksize)
    ax1.set_xlim(0.0, 2.5)
    ax1.set_ylim(0.0, 1.0)
    ax1.legend([r"$N = {}$".format(int(n / 2)) for n in N], bbox_to_anchor = (0.05, 0.4, 0.25, 0.2), loc="lower left", mode = "expand", ncol = 1)
    plt.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Transfer_Report" + "/R_k_TIME_N_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()







    fig = plt.figure(figsize = (3.4, 2.4), tight_layout = False)
    gs  = GridSpec(1, 1)   


    labelsize = 8
    ticksize  = 6

    ax1 = fig.add_subplot(gs[0, 0])
    # ax1.set_prop_cycle(cycler('color', ['b', 'g',  'r', 'c', 'm', 'y', 'k']))
    # for j in range(len(N)):
    ax1.plot(alpha, R_k_N_alpha[-1, :], '-', linewidth = 0.75)
        # ax1.plot(alpha, R_k_N_alpha_half[j, :], '--', linewidth = 0.75)
    ax1.legend([r"Order"])
    ax1.set_xlabel(r"$\alpha$", fontsize = labelsize)
    ax1.set_ylabel(r"$\mathcal{R}$", fontsize = labelsize)   
    # ax1.legend([r"$N = {}$".format(int(n / 2)) for n in N], fontsize = labelsize)
    ax1.tick_params(labelsize = ticksize)
    ax1.set_xlim(0.0, 2.5)
    ax1.set_ylim(0.0, 1.0)
    plt.savefig(output_dir + "/R_k_TIME_N_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()

    fig = plt.figure(figsize = (3.4, 2.4), tight_layout = False)
    gs  = GridSpec(1, 1)   


    labelsize = 8
    ticksize  = 6

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_prop_cycle(cycler('color', ['b', 'g',  'r', 'c', 'm', 'y', 'k']))
    for j in range(len(N)):
        ax1.plot(alpha, R_k_N_alpha[j, :], '-', linewidth = 0.75)
        # ax1.plot(alpha, R_k_N_alpha_half[j, :], '--', linewidth = 0.75)
    ax1.legend([r"Order"])
    ax1.set_xlabel(r"$\alpha$", fontsize = labelsize)
    ax1.set_ylabel(r"$\mathcal{R}$", fontsize = labelsize)   
    ax1.legend([r"$N = {}$".format(int(n / 2)) for n in N], fontsize = labelsize)
    ax1.tick_params(labelsize = ticksize)
    ax1.set_xlim(0.0, 2.5)
    for j in range(len(N)):
        ax1.plot(alpha, R_k_N_alpha_half[j, :], '--', linewidth = 0.75)
    ax1.legend([r"$N = {}$".format(int(n / 2)) for n in N], fontsize = labelsize)
    ax1.set_ylim(0.0, 1.0)
    plt.savefig(output_dir + "/R_k_TIME_N_both_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()

    fig = plt.figure(figsize = (3.4, 2.4), tight_layout = False)
    gs  = GridSpec(1, 1)   

    ax1 = fig.add_subplot(gs[0, 0])
    for j in range(len(N)):
        ax1.plot(alpha, R_k_N_alpha_cent[j, :], '-', linewidth = 0.75)
    ax1.legend([r"Order"])
    ax1.set_xlabel(r"$\alpha$", fontsize = labelsize)
    ax1.set_ylabel(r"$\mathcal{R}$", fontsize = labelsize)   
    ax1.legend([r"$N = {}$".format(int(n / 2)) for n in N], fontsize = labelsize)
    ax1.tick_params(labelsize = ticksize)
    ax1.set_xlim(0.0, 2.5)
    # ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Centroid")
    plt.savefig(output_dir + "/R_k_TIME_N_centrd_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()
    fig = plt.figure(figsize = (3.4, 2.4), tight_layout = False)
    gs  = GridSpec(1, 1)   


    ax1 = fig.add_subplot(gs[0, 0])
    # for j in range(len(N)):
    ax1.plot(alpha, R_k_N_alpha_half[-1, :], '-', linewidth = 0.75)
    ax1.legend([r"Order"])
    ax1.set_xlabel(r"$\alpha$", fontsize = labelsize)
    ax1.set_ylabel(r"$\mathcal{R}$", fontsize = labelsize)   
    # ax1.legend([r"$N = {}$".format(int(n / 2)) for n in N], fontsize = labelsize)
    ax1.tick_params(labelsize = ticksize)
    ax1.set_xlim(0.0, 2.5)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("N/2")
    plt.savefig(output_dir + "/R_k_TIME_N_half_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()

    fig = plt.figure(figsize = (3.4, 2.4), tight_layout = False)
    gs  = GridSpec(1, 1)   

    ax1 = fig.add_subplot(gs[0, 0])
    for j in range(len(N)):
        ax1.plot(alpha, R_k_N_alpha_half_cent[j, :], '-', linewidth = 0.75)
    ax1.legend([r"Order"])
    ax1.set_xlabel(r"$\alpha$", fontsize = labelsize)
    ax1.set_ylabel(r"$\mathcal{R}$", fontsize = labelsize)   
    ax1.legend([r"$N = {}$".format(int(n / 2)) for n in N], fontsize = labelsize)
    ax1.tick_params(labelsize = ticksize)
    ax1.set_xlim(0.0, 2.5)
    # ax1.set_ylim(0.0, 1.0)
    ax1.set_title("N/2 and Centroid")
    plt.savefig(output_dir + "/R_k_TIME_N_half_centrd_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()




    #######################
    ##     Plot Data     ##
    #######################
    # N = [256]

    # krange = np.arange(5, 80 + 1)

    # colours = cm.jet(np.linspace(0, 1, len(alpha)))
    
    # slope_amp = np.zeros((len(N), len(alpha)))
    # slope_p_k = np.zeros((len(N), len(alpha)))

    # fig = plt.figure(figsize = (16, 9), tight_layout = False)
    # gs  = GridSpec(2, 2)  

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, 0])
    # ax4 = fig.add_subplot(gs[1, 1])


    # fig1 = plt.figure(figsize = (16, 9), tight_layout = False)
    # gs1  = GridSpec(1, 1)  

    # ax11 = fig.add_subplot(gs1[0, 0])
    # for j, n in enumerate(N):
    #     for i, a in enumerate(alpha):

    #         ## Get Filename
    #         filename      = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, trans)
    #         filename10    = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/10))
    #         filename100   = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/100))
    #         filename1000  = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/1000))
    #         filename10000 = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/10000))

          
           
    #         ## Open in current file
    #         if os.path.exists(input_dir + filename + '.h5'):
    #             HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')
    #         elif os.path.exists(input_dir + filename10 + '.h5'):
    #             HDFfileData = h5py.File(input_dir + filename10 + '.h5', 'r')
    #         elif os.path.exists(input_dir + filename100 + '.h5'):
    #             HDFfileData = h5py.File(input_dir + filename100 + '.h5', 'r')
    #         elif os.path.exists(input_dir + filename1000 + '.h5'):
    #             HDFfileData = h5py.File(input_dir + filename1000 + '.h5', 'r')
    #         elif os.path.exists(input_dir + filename1000 + '.h5'):
    #             HDFfileData = h5py.File(input_dir + filename1000 + '.h5', 'r')
    #         else:
    #             print("File doesn't exist!...Alpha = {:.3f}".format(a))
    #             continue
            
    #         ## Read in data
    #         # Amplitudes
    #         a_k     = HDFfileData['Amps'][:]
    #         num_osc = a_k.shape[0]
    #         kmin    = k0 + 1
    #         kmax    = num_osc - 1
    #         P_k = HDFfileData['P_k_avg']

    #         ## Compute the amplitude normalization factor
    #         amp_norm = amp_normalization(a_k, num_osc, k0)

    #         ## Do a line fit 
    #         slope_amp[j, i] = (np.log(amp_norm[krange[-1]]) - np.log(amp_norm[krange[0]])) / (np.log(krange[-1]) - np.log(krange[0]))
    #         slope_p_k[j, i] = (np.log(P_k[krange[-1]]) - np.log(P_k[krange[0]])) / (np.log(krange[-1]) - np.log(krange[0]))

    #         print("a = {:0.3f} || slope = {}".format(a, slope_amp[j, i]))

    #         ax1.plot(np.arange(kmin, kmax + 1), amp_norm[kmin:], '-', color = colours[i], lw = 0.5)
    #         ax1.plot(krange, (krange ** slope_amp[j, i]) * (amp_norm[5:80 + 1] * (1/ (krange ** slope_amp[j, i]))), '--', color = colours[i])
    #         ax1.set_xlabel(r"$k$")
    #         ax1.set_ylabel(r"$\sum_{k_1}a_{k_1}a_{k - k_1}$")
    #         ax1.set_xscale('log')
    #         ax1.set_yscale('log')

    #         ax3.plot(np.arange(kmin, kmax + 1), P_k[kmin:], '-', color = colours[i], lw = 0.5)
    #         ax3.plot(krange, (krange ** slope_p_k[j, i]) , '--', color = colours[i])
    #         ax3.set_xlabel(r"$k$")
    #         ax3.set_ylabel(r"$P_k$")
    #         ax3.set_xscale('log')
    #         ax3.set_yscale('log')

    #         HDFfileData.close()

        

    # ax1.legend([x for x in itertools.chain.from_iterable(itertools.zip_longest([r"$\alpha =$ {:0.1f}".format(a) for a in alpha], [r"Slope = {:.01f}".format(s) for s in slope_amp[0, :]])) if x], ncol = 2, bbox_to_anchor=(-0.1, -0.5), loc="lower right")
    # # ax1.legend([r"$\alpha =$ {:0.1f}".format(a) for a in alpha], bbox_to_anchor=(-0.1, 0.2), loc="lower right")
    
    # for j in range(len(N)):
    #     ax2.plot(alpha, slope_amp[j, :])
    # ax2.set_xlabel(r"$\alpha$")
    # ax2.set_ylabel(r"SLOPE")
    # ax2.set_title(r"Amp Norm")

    # for j in range(len(N)):
    #     ax4.plot(alpha, slope_p_k[j, :])
    # ax4.set_xlabel(r"$\alpha$")
    # ax4.set_ylabel(r"SLOPE")
    # ax4.set_title(r"$P_k$")

    # # ax2.legend([r"$N = $ {}".format(n) for n in N], bbox_to_anchor=(-0.1, 0.2), loc="lower right")


    # # ax2.legend([r"$\alpha =$ {:0.1f}".format(a) for a in alpha], bbox_to_anchor=(1.04,1), loc="upper left")
    # # plt.savefig(output_dir + "/AmpNorm_Slope_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(beta, k0, iters, u0), bbox_inches='tight') 
    
    # plt.savefig(output_dir + "/AmpNorm_Slope_N[{}]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(n, beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()







    ## 
    # n = 2048
    # a = 1.7
    # filename      = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, trans)
    # HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')
    # R_k_short = HDFfileData['R_k_avg']

    # filename      = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, int(1e7), trans)
    # HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')
    # R_k_long = HDFfileData['R_k_avg']


    # fig = plt.figure(figsize = (16, 9), tight_layout = False)
    # gs  = GridSpec(1, 1)  

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.plot(R_k_short, '-')
    # ax1.plot(R_k_long, '-')
    # ax1.set_xlabel(r"$k$")
    # ax1.set_ylabel(r"$R_k$")
    # ax1.legend([r"Iters = 4e5", r"Iters = 1e7"])

    # plt.savefig(output_dir + "/R_k_compare_N[{}]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(n, beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()






    ####
    # n     = 2048
    # alpha = [0.0, 0.5, 1.0, 1.2, 1.3, 1.5, 2.0, 2.5]



    # fig1 = plt.figure(figsize = (16, 9), tight_layout = False)
    # gs1  = GridSpec(2, 4) 
    # fig2 = plt.figure(figsize = (16, 9), tight_layout = False)
    # gs2  = GridSpec(2, 4) 

    # ax1 = []
    # ax2 = []
    # for i in range(2):
    #     for j in range(4):
    #         ax1.append(fig1.add_subplot(gs1[i, j]))
    #         ax2.append(fig2.add_subplot(gs2[i, j]))

    # for i, a in enumerate(alpha):

    #     print("a = {}".format(a))

    #     ## Get Filename
    #     filename      = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, trans)
    #     filename10    = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/10))
    #     filename100   = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/100))
    #     filename1000  = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/1000))
    #     filename10000 = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/10000))

                
    #     ## Open in current file
    #     if os.path.exists(input_dir + filename + '.h5'):
    #         HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')
    #     elif os.path.exists(input_dir + filename10 + '.h5'):
    #         HDFfileData = h5py.File(input_dir + filename10 + '.h5', 'r')
    #     elif os.path.exists(input_dir + filename100 + '.h5'):
    #         HDFfileData = h5py.File(input_dir + filename100 + '.h5', 'r')
    #     elif os.path.exists(input_dir + filename1000 + '.h5'):
    #         HDFfileData = h5py.File(input_dir + filename1000 + '.h5', 'r')
    #     elif os.path.exists(input_dir + filename1000 + '.h5'):
    #         HDFfileData = h5py.File(input_dir + filename1000 + '.h5', 'r')
    #     else:
    #         print("File doesn't exist!...Alpha = {:.3f}".format(a))
    #         continue
        
    #     ## Read in data
    #     # Amplitudes
    #     a_k     = HDFfileData['Amps'][:]
    #     num_osc = a_k.shape[0]
    #     kmin    = k0 + 1
    #     kmax    = num_osc - 1
    #     sin_theta_k_avg = HDFfileData['SinTheta_k_avg'][:]

    #     # P_k
    #     P_k_avg = HDFfileData['P_k_avg'][:]

    #     ax1[i].plot(np.arange(kmin, kmax + 1), sin_theta_k_avg[kmin:], '-')
    #     ax1[i].set_xlabel(r"$k$")
    #     ax1[i].set_ylabel(r"$\langle\sin\theta_k\rangle_t$")
    #     ax1[i].legend([r"$\alpha = {}$".format(a)])
    #     ax1[i].set_xscale('log')

    #     ax2[i].plot(np.arange(kmin, kmax + 1), P_k_avg[kmin:], '-')
    #     ax2[i].set_xlabel(r"$k$")
    #     ax2[i].set_ylabel(r"$\langle P_{k} \rangle_t$")
    #     ax2[i].legend([r"$\alpha = {}$".format(a)])
    #     ax2[i].set_yscale('log')
    #     ax2[i].set_xscale('log')

    #     HDFfileData.close()

    # # ax11.legend([r"$\alpha = ${}".format(a) for a in alpha])
    # fig1.savefig(output_dir + "/SIN_THETA_k_N[{}]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(n, beta, k0, iters, u0), bbox_inches='tight') 
  

    # fig2.savefig(output_dir + "/aaaaa_avg_N[{}]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(n, beta, k0, iters, u0), bbox_inches='tight') 
  
    # plt.close()
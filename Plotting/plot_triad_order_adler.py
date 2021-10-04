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
mpl.rcParams['figure.figsize']    = [10, 8]
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['text.usetex']       = True
mpl.rcParams['font.family']       = 'serif'
mpl.rcParams['font.serif']        = 'Computer Modern Roman'
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import h5py
import sys
import os
import time as TIME
import multiprocessing as mprocs
from threading import Thread
from subprocess import Popen, PIPE
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import zip_longest
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm 
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from numba import jit, njit

from scipy.interpolate import interp1d


#########################
##    Function Defs    ##
#########################
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

def convolution_fft(u_z, N, num_osc, k0):

    M = 2 * N
    norm_fac = 1 / M

    conv = np.ones((num_osc, )) * np.complex(0.0, 0.0)
        
    u_tmp = (np.real(np.fft.ifft(u_z, n = M) * 2*M)) ** 2

    u_z_tmp = np.fft.fft(u_tmp, n= M)
    
    for i in range(num_osc):
        if i <= k0:
            conv[i] = np.complex(0.0, 0.0)
        else:
            conv[i] = u_z_tmp[i] * norm_fac

    return conv

@njit
def compute_F_k(triad_order_k, amps, k0, num_osc):

    F_k = np.zeros((triad_order_k.shape))

    for k in range(num_osc):
        if k <= k0: 
            F_k[:, k] = 0.0
        else:
            F_k[:, k] = (k / 2 * amps[k]) * np.absolute(triad_order_k[:, k])

    return F_k


@njit
def compute_dPhi_k_dt(Phi_k, num_osc, kmin):

    dPhi_k = np.zeros(Phi_k.shape)
    tmp    = np.zeros(Phi_k.shape[1])

    norm = 1
    for t in range(1, Phi_k.shape[0]):
        tmp[:]       += Phi_k[t, :] - Phi_k[t - 1, :]
        dPhi_k[t, :] = tmp / norm
        norm         += 1

    return dPhi_k


@njit 
def compute_tau_k(F_k, time, num_osc, kmin, k):

    tau_k = np.zeros((F_k.shape[0]))

    numT = F_k.shape[0]

    for t in range(1, numT):
        # for k in range(kmin, num_osc):
        tau_k[t] = np.trapz(F_k[:t + 1, k], x = time[:t + 1])

    return tau_k

@njit
def compute_omega_k_tseries(F_k, dPhi_k, num_osc, kmin):

    omega_k = np.zeros(dPhi_k.shape)

    norm = 1
    tmp1 = np.zeros(dPhi_k.shape[1])
    tmp2 = np.zeros(dPhi_k.shape[1])

    for t in range(dPhi_k.shape[0]):
        tmp1 += dPhi_k[t, :]
        tmp2 += F_k[t, :]

        omega_k[t, :] = (tmp1[:] / norm) / (tmp2[:] / norm)

        norm += 1

    return omega_k

@njit
def compute_omega_k_tseries_window(F_k, dPhi_k, num_osc, kmin, win_size):

    omega_k = np.zeros(dPhi_k.shape)

    for t in range(dPhi_k.shape[0]):
        if t >= win_size:
            for k in range(dPhi_k.shape[1]):
                omega_k[t, k] = np.mean(dPhi_k[t - win_size:t, k]) / np.mean(F_k[t - win_size:t, k])

    return omega_k


def compute_omega_k_tseries_window_alt(F_k, dPhi_k, num_osc, kmin, win_size):

    # omega_k = np.zeros((F_k.shape[0] - win_size, num_osc))
    # omega_k = np.zeros(dPhi_k.shape)

    # for k in range(kmin, dPhi_k.shape[1]):
    #     tmp1      = np.convolve(dPhi_k[:, k], np.ones(win_size) / win_size, mode = 'valid')
    #     tmp2      = np.convolve(F_k[1:, k], np.ones(win_size) / win_size, mode = 'valid')
    #     omega_k[:, k] = tmp1 / tmp2

    cumsum   = np.zeros(dPhi_k.shape)
    cumsum_1 = np.zeros(dPhi_k.shape)

    for k in range(F_k.shape[1]):
        cumsum[:, k]   = np.cumsum(F_k[1:, k])
        cumsum_1[:, k] = np.cumsum(dPhi_k[:, k])

    omega_k = ((cumsum_1[win_size:, :] - cumsum_1[:-win_size, :])) / ((cumsum[win_size:, :] - cumsum[:-win_size, :]))
    return omega_k




@njit
def compute_clv_stats_data(clv, a_k, num_tsteps, dof, numLEs):
    
    ## Memory Allocation
    v_k      = np.zeros((dof, dof))
    p_k      = np.zeros((dof, dof))
    v_k_proj = np.zeros((dof, dof))

    ## Translation Invariant Direction -> T
    T           = np.arange(2.0, float(dof + 2), 1.0)
    T_a_k       = T * a_k[2:]
    T_norm_sqr  = np.linalg.norm(T) ** 2
    T_enorm_sqr = np.linalg.norm(T_a_k) ** 2
    
    ## Loop over time
    for t in range(num_tsteps):

        ## Loop over vectors
        for j in range(numLEs):
            
            ## Square each component
            v_k[:, j] += np.square(clv[t, :, j])

            ## Compute the projection
            v_proj  = clv[t, :, j] - (T * (np.dot(clv[t, :, j], T))) / T_norm_sqr
            clv_a_k = clv[t, :, j] * a_k[2:]
            v_enorm = clv_a_k - (T_a_k * np.dot(clv_a_k, T_a_k)) / T_enorm_sqr
            
            ## Renormalize after projection
            v_proj  = v_proj / np.linalg.norm(v_proj)
            v_enorm = v_enorm / np.linalg.norm(v_enorm)
            
            ## Update running sum
            p_k[:, j]      += np.square(v_enorm)
            v_k_proj[:, j] += np.square(v_proj)
            
    ## Compute averages
    v_k      = v_k / num_tsteps
    p_k      = p_k / num_tsteps
    v_k_proj = v_k_proj / num_tsteps
    

    return v_k, v_k_proj, p_k


if __name__ == '__main__':
    #########################
    ##  Get Input Parameters
    #########################
    if (len(sys.argv) != 8):
        print("No Input Provided, Error.\nProvide k0\nAlpah\nBeta\nIterations\nTransient Iterations\nN\nu0\n")
        sys.exit()
    else: 
        k0    = int(sys.argv[1])
        alpha = float(sys.argv[2])
        beta  = float(sys.argv[3])
        iters = int(sys.argv[4])
        trans = int(sys.argv[5])
        N     = int(sys.argv[6])
        u0    = str(sys.argv[7])
    results_dir = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]".format(N, k0, alpha, beta, u0)
    filename    = "/SolverData_ITERS[{}]_TRANS[{}]".format(iters, trans)

    ######################
    ##  Input & Output Dir
    ######################
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/PhaseOrder"




    ######################
    ##  Read in Input File
    ######################
    HDFfileData = h5py.File(input_dir + results_dir + filename + '.h5', 'r')

    # print input file name to screen
    print("\n\nData File: {}.h5\n".format(results_dir + filename))

    # print(list(HDFfileData.keys()))
    ######################
    ##  Read in Datasets
    ######################
    # phi    = HDFfileData['Phases'][:, :]
    time   = HDFfileData['Time'][:]
    amps   = HDFfileData['Amps'][:]




    # ######################
    # ##  Preliminary Calcs
    # ######################
    ntsteps = len(time)
    num_osc = amps.shape[0]
    N       = 2 * (num_osc - 1)
    kmin    = k0 + 1
    kmax    = num_osc - 1




    ######################
    ##  Plot Data
    ######################
    Adler_order_k = HDFfileData['AdlerScaleOrderParam'][:, :]
    P_k = np.mean(np.absolute(Adler_order_k[:, kmin:]), axis = 0)

    # Normalization factor
    amp_norm = amp_normalization(amps, num_osc, k0)

    fig = plt.figure(figsize = (16, 9), tight_layout = False)
    gs  = GridSpec(1, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.arange(kmin, kmax + 1), amp_norm[k0 + 1:], '-')
    # ax1.plot(np.arange(kmin, kmax + 1), np.convolve(amp_norm[k0 + 1:], amp_norm[k0 + 1:], mode = 'valid'))
    ax1.plot(np.arange(kmin, kmax + 1), 1 / np.power(np.arange(kmin, kmax+ 1), 2), '--', color='b')
    ax1.plot(np.arange(kmin, kmax + 1), 1 / np.sqrt(np.arange(kmin, kmax+ 1)) , '-', color = 'r')
    ax1.plot(np.arange(kmin, kmax + 1), P_k, '--', color='k')
    ax1.plot(np.arange(kmin, kmax + 1), P_k / amp_norm[k0 + 1:] , '-')
    ax1.set_title(r"Amplitude Normalization Factor")
    ax1.set_xlabel(r"$k$")
    ax1.set_ylabel(r"$\sum_{k_1}a_{k_1, k - k_1}$")
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend([r"Norm", r"$k^{-2}$", r"$k^{-1/2}$", r"$P_k$", r"Ratio"])
    plt.savefig(output_dir + "/AMP_NORM_A_N[{}]_ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N, alpha, beta, k0, iters, u0), bbox_inches='tight') 
    plt.close() 
    

    ## F_k
    # adler_order_k = HDFfileData['AdlerScaleOrderParam'][:, :]
   
    # F_k = compute_F_k(adler_order_k, amps, k0, num_osc)    

    # fig = plt.figure(figsize = (16, 9), tight_layout = False)
    # gs  = GridSpec(1, 1)
    
    # krange = range(2, 16)


    # ax1 = fig.add_subplot(gs[0, 0])
    # for k in krange:
    #     ax1.plot(F_k[:, k], '-')
    # ax1.set_title(r"$F_k$ Time Series")
    # ax1.set_xlabel(r"$t$")
    # ax1.set_ylabel(r"$F_k$")
    # ax1.legend([r"$k = {}$".format(k) for k in krange])
    # ax1.set_yscale('log')

    # plt.savefig(output_dir + "/F_k_ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(alpha, beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close() 


    # fig = plt.figure(figsize = (36, 16), tight_layout = False)
    # gs  = GridSpec(4, 4, hspace = 0.2)
    
    # ax = []
    # for i in range(4):
    #     for j in range(4):
    #         ax.append(fig.add_subplot(gs[i, j]))
    
    # for i in range(16):
    #     ax[i].plot(time[:], np.angle(adler_order_k[:, i]), '-')
    #     ax[i].set_title(r"$\Phi_k(t), k = {}$".format(k))
    #     # ax[i].set_ylabel(r"$\theta_k(t)$")
    #     ax[i].set_xlabel(r"t")  

    # plt.savefig(output_dir + "/PHI_k_TSERIES_PDF_N[{}]_ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N, alpha, beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close() 


    # #####################################################
    # ##     Kuramoto Time Order Parameter - Theta       ##
    # #####################################################
    # Theta Order Parameter
    theta_order_k      = HDFfileData['ThetaTimeScaleOrderParam'][:, :]
    PhaseShift_order_k = HDFfileData['PhaseShiftScaleOrderParam'][:, :]

    
    # fig = plt.figure(figsize = (32, 9), tight_layout = False)
    # gs  = GridSpec(1, 3)

    # krange = range(2, 16)#[k0 + 1, 4, 6, 8, 10, 12, 20, 30]
    
    # ax1 = fig.add_subplot(gs[0, 0])
    # for k in krange:
    #     ax1.plot(np.absolute(theta_order_k[:, k]), '-') # np.cos(np.angle(PhaseShift_order_k[:, k]))
    # ax1.set_title(r"Theta Time Order Time Series")
    # ax1.set_xlabel(r"$t$")
    # ax1.set_ylabel(r"$R_k$")
    # ax1.legend([r"$k = {}$".format(k) for k in krange])

    # ax2 = fig.add_subplot(gs[0, 1])
    # ax2.plot(range(k0 + 1, num_osc), np.absolute(theta_order_k[-1, k0 +1:]), '-')
    # ax2.set_title(r"Theta Time Order R_k")
    # ax2.set_xlabel(r"$k$")
    # ax2.set_ylabel(r"$R_k$")
    # # ax2.legend([r"$k = {}$".format(k) for k in krange])

    # ax2 = fig.add_subplot(gs[0, 2])
    # for k in krange:
    #     ax2.plot(np.cos(np.angle(theta_order_k[:, k])), '-')
    # ax2.set_title(r"Theta Time Order Time Series")
    # ax2.set_xlabel(r"$t$")
    # ax2.set_ylabel(r"$\Phi_k$")
    # ax2.legend([r"$k = {}$".format(k) for k in krange])



    fig = plt.figure(figsize = (32, 9), tight_layout = False)
    gs  = GridSpec(1, 1)

    ax2 = fig.add_subplot(gs[0, 0])
    ax2.plot(range(k0 + 1, num_osc), np.absolute(theta_order_k[-1, k0 +1:]), '-')
    ax2.set_title(r"Theta Time Order $R_k$")
    ax2.set_xlabel(r"$k$")
    ax2.set_ylabel(r"$R_k$")
    ax2.set_xscale('log')

    plt.savefig(output_dir + "/TIME_ORDER_THETA_N_[{}]_ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N, alpha, beta, k0, iters, u0), bbox_inches='tight') 
    plt.close() 



    #######################
    ##      Task 3       ##
    #######################
    # Phase Locking - T-series
    # Adler_order_k = HDFfileData['AdlerScaleOrderParam'][:, :]

    # F_k  = (np.array(range(kmin, num_osc), dtype = 'int64') / (2 * amps[kmin:])) * np.absolute(Adler_order_k[:, kmin:])

    # P_k = np.absolute(Adler_order_k[:, kmin:])

    # P_k = P_k / np.amax(P_k, axis = 0)

    # dPhi_k = np.angle(Adler_order_k[1:, kmin:] * np.conjugate(Adler_order_k[:-1, kmin:])) / (time[1] - time[0])
    
    # krange = range(0, 16 - kmin)

    # for kk in krange:
    #     fig = plt.figure(figsize = (32, 9), tight_layout = False)
    #     gs  = GridSpec(1, 2)

    #     ax1 = fig.add_subplot(gs[0, 0])
    #     ax1.plot(time[1:], dPhi_k[:, kk] / F_k[1:, kk])
    #     ax1.set_title(r"$\frac{\mathrm{d}\Phi_k (t(\tau_k))}{\mathrm{d}\tau_k}$ Time Series")
    #     ax1.set_xlabel(r"$t$")
    #     ax1.set_ylabel(r"$\frac{\mathrm{d}\Phi_k (t(\tau_k))}{\mathrm{d}\tau_k}$")
    #     ax1.legend([r"$k = {}$".format(kk)])
    #     plt.savefig(output_dir + "/TASK3_TSERIES_k[{}]_N[{}]_ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(kk, N, alpha, beta, k0, iters, u0), bbox_inches='tight') 
    #     plt.close() 

    #######################
    ##      Task 4       ##
    #######################
    # # Phase Locking - T-series
    # Adler_order_k = HDFfileData['AdlerScaleOrderParam'][:, :]

    # F_k  = (np.array(range(kmin, num_osc), dtype = 'int64') / (2 * amps[kmin:])) * np.absolute(Adler_order_k[:, kmin:])
    
    # krange = range(0, 16 - kmin)

    # dPhi_k = np.angle(Adler_order_k[1:, kmin:] * np.conjugate(Adler_order_k[:-1, kmin:])) / (time[1] - time[0])


    # for i in [0, 1, 2, 5, 10, 25, 50, 100, 150, 300]:
    #     # Omega_k = compute_omega_k_tseries(F_k, dPhi_k, num_osc, kmin)
    #     Omega_k = compute_omega_k_tseries_window_alt(F_k, dPhi_k, num_osc, kmin, i)

    #     fig = plt.figure(figsize = (32, 9), tight_layout = False)
    #     gs  = GridSpec(1, 2)

    #     ax1 = fig.add_subplot(gs[0, 0])
    #     for k in krange:
    #         ax1.plot(Omega_k[:, k], '-') 
    #     ax1.set_title(r"Phase Locking: $\Omega_k$ Time Series")
    #     ax1.set_xlabel(r"$t$")
    #     ax1.set_ylabel(r"$\Omega_k$")
    #     ax1.legend([r"$k = {}$".format(k) for k in krange])
    #     # ax1.set_ylim(-0.1, 0.1)

    #     ax2 = fig.add_subplot(gs[0, 1])
    #     for k in krange:
    #         ax2.plot(Omega_k[:, int(k + krange[-1])], '-') 
    #     ax2.set_title(r"Phase Locking: $\Omega_k$ Time Series")
    #     ax2.set_xlabel(r"$t$")
    #     ax2.set_ylabel(r"$\Omega_k$")
    #     ax2.legend([r"$k = {}$".format(int(k + krange[-1])) for k in krange])
    #     # ax2.set_ylim(-0.1, 0.1)
        

    #     plt.savefig(output_dir + "/TASK4_OMEGA_k_CONVERGENCE_TSERIES_WINDOW[{}]_N[{}]_ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(i, N, alpha, beta, k0, iters, u0), bbox_inches='tight') 
    #     plt.close() 

   


    # #######################
    # ##      Task 5       ##
    # #######################
    # ## Omega_k over time and PDF
    # fig = plt.figure(figsize = (32, 9), tight_layout = False)
    # gs  = GridSpec(1, 3)

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax3 = fig.add_subplot(gs[0, 1])
    # ax2 = fig.add_subplot(gs[0, 2])
    # for k in krange:
    #     k = k        
    #     ax1.plot(Omega_k[:, k], '-') 
    #     ax1.set_title(r"Phase Locking: $\Omega_k$ Time Series")
    #     ax1.set_xlabel(r"$t$")
    #     ax1.set_ylabel(r"$\Omega_k$")
    #     ax1.legend([r"$k =$ {}".format(i) for i in krange])
    #     ax1.set_ylim(-100, 100)
        
    #     hist, bins = np.histogram(Omega_k[-100:, k], bins = 1000, density = False)
    #     bin_centres     = (bins[1:] + bins[:-1]) * 0.5
    #     bin_width       = (bins[1] - bins[0])
    #     pdf = hist / (np.sum(hist) * bin_width) 
    #     ax3.plot(bin_centres, pdf, '.-')  
    #     ax3.set_title(r"PDF of $\Omega_k$: Last 100 Iterations")
    #     ax3.set_xlabel(r"$\Omega_k$")
    #     ax3.set_ylabel(r"PDF")
    #     ax3.legend([r"$k =$ {}".format(i) for i in krange])

    #     hist, bins = np.histogram(Omega_k[:, k], bins = 1000, density = False)
    #     bin_centres     = (bins[1:] + bins[:-1]) * 0.5
    #     bin_width       = (bins[1] - bins[0])
    #     pdf = hist / (np.sum(hist) * bin_width) 
    #     ax2.plot(bin_centres, pdf, '.-')  
    #     ax2.set_title(r"PDF of $\Omega_k$")
    #     ax2.set_xlabel(r"$\Omega_k$")
    #     ax2.set_ylabel(r"PDF")
    #     ax2.set_xlim(-10, 10)
    #     ax2.legend([r"$k =$ {}".format(i) for i in krange])
 
    
    # plt.savefig(output_dir + "/TASK5_1_OMEGA_k_N[{}]_ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N, alpha, beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close() 



    # ## Phase Locking
    # Omega_k = HDFfileData['PhiPhaseLocking'][:]

    # Omega_k_py = compute_omega_k_tseries(F_k, dPhi_k, num_osc, kmin)

    # fig = plt.figure(figsize = (32, 9), tight_layout = False)
    # gs  = GridSpec(2, 2)

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.plot(Omega_k[:], '-') 
    # ax1.plot(Omega_k_py[-1, :], '-') 
    # ax1.set_title(r"Phase Locking: $\Omega_k$")
    # ax1.set_xlabel(r"$k$")
    # ax1.set_ylabel(r"$\Omega_k$")
    # # ax1.set_ylim(-1, 1)

    # ax2 = fig.add_subplot(gs[0, 1])
    # counts, bins = np.histogram(Omega_k[:], density = False) 
    # bin_cents    = (bins[1:] + bins[:-1]) * 0.5
    # bin_widths   = bins[1] - bins[0]
    # pdf = counts / (np.sum(counts) * bin_widths)
    # ax2.plot(bin_cents, pdf)
    # ax2.set_title(r"PDF of $\Omega_k$")
    # ax2.set_xlabel(r"$\Omega_k$")
    # ax2.set_ylabel(r"$PDF$")

    # kx = np.array(range(k0 + 1, num_osc), dtype = 'int64')

    # locked   = np.absolute(Omega_k[-1, :]) < 1
    # unlocked = np.absolute(Omega_k[-1, :]) > 1

    # locked_k   = kx[locked]
    # unlocked_k = kx[unlocked]

    # print(len(locked_k))
    # print(len(unlocked_k))

    # ax3 = fig.add_subplot(gs[1, 0])
    # ax3.plot(range(k0 + 1, num_osc), locked, '*')
    # ax3.set_title(r"Locked")
    # ax3.set_xlabel(r"$k$")
    # ax3.set_ylabel(r"$Unlocked$")
    

    # ax4 = fig.add_subplot(gs[1, 1])
    # ax4.plot(range(k0 + 1, num_osc), unlocked, '*')
    # ax4.set_title(r"Unlocked")
    # ax4.set_xlabel(r"$k$")
    # ax4.set_ylabel(r"$Unlocked$")

    # plt.savefig(output_dir + "/TASK5_2_OMEGA_k_N[{}]_ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N, alpha, beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close() 



    #######################
    ##      Task 6       ##
    #######################
    ## Phase Locking Theta_k
    # theta_k = HDFfileData['Theta_k'][:, :]

    # fig = plt.figure(figsize = (36, 16), tight_layout = False)
    # gs  = GridSpec(4, 4, hspace = 0.2)
    
    # ax = []
    # for i in range(4):
    #     for j in range(4):
    #         ax.append(fig.add_subplot(gs[i, j]))
    # print(theta_k.shape)
    # for i in range(16):
    #     kk = i + 55
    #     # print(np.arcsin(Omega_k[locked_k[i]]))
    #     ax[i].plot(time[:], np.cos(theta_k[:, kk]), '-')
    #     # ax[i].axhline(y = np.arcsin(Omega_k[locked_k[i]]), xmin = time[0], xmax = time[-1], color = 'k', linestyle = '--')
    #     ax[i].set_title(r"PDF of $\theta_k(t) Locked: k = {}$".format(kk))
    #     ax[i].set_ylabel(r"$\theta_k(t)$")
    #     ax[i].set_xlabel(r"t")  

    # plt.savefig(output_dir + "/TASK6_1_THETA_k_TSERIES_LOCKED_PDF_N[{}]_ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N, alpha, beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close() 

    # fig = plt.figure(figsize = (36, 16), tight_layout = False)
    # gs  = GridSpec(4, 4, hspace = 0.2)
    
    # ax = []
    # for i in range(4):
    #     for j in range(4):
    #         ax.append(fig.add_subplot(gs[i, j]))
    
    # for i in range(16):
    #     ax[i].plot(time[:], theta_k[:, unlocked_k[i]], '-')
    #     ax[i].set_title(r"PDF of $\theta_k(t) Unlocked: k = {}$".format(unlocked_k[i]))
    #     ax[i].set_ylabel(r"$\theta_k(t)$")
    #     ax[i].set_xlabel(r"t")  

    # plt.savefig(output_dir + "/TASK6_2_THETA_k_TSERIES_UNLOCKED_PDF_N[{}]_ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N, alpha, beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close() 

    # fig = plt.figure(figsize = (36, 16), tight_layout = False)
    # gs  = GridSpec(4, 4, hspace = 0.2)
    
    # ax = []
    # for i in range(4):
    #     for j in range(4):
    #         ax.append(fig.add_subplot(gs[i, j]))
    
    # for i in range(16):

    #     counts, bins = np.histogram(theta_k[:, i], bins = 1000, density = False) 
    #     bin_cents    = (bins[1:] + bins[:-1]) * 0.5
    #     bin_width    = bins[1] - bins[0]
    #     pdf = counts / (np.sum(counts) * bin_width)
    #     ax[i].plot(bin_cents, pdf, '-')
    #     ax[i].set_yscale('log')

    #     # ax[i].axvline(x = np.arcsin(Omega_k[locked_k[i]]), ymin = 0.0, ymax = 1, color = 'k', linestyle = '--')
    #     ax[i].set_title(r"PDF of $\theta_k(t) Locked: k = {}$".format(i))
    #     ax[i].set_xlabel(r"$\theta_k(t)$")
    #     ax[i].set_ylabel(r"PDF")  

    # plt.savefig(output_dir + "/TASK6_3_THETA_k_LOCKED_PDF_N[{}]_ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N, alpha, beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close() 


    # #######################
    # ##      Task 7       ##
    # #######################
    # Adler_order_k = HDFfileData['AdlerScaleOrderParam'][:, :]

    # k_unlock = unlocked_k[5]
    
    # F_k     = (np.array(range(0, num_osc), dtype = 'int64') / 2 * amps) * np.absolute(Adler_order_k[:, :])
    # F_k_avg = np.mean(F_k, axis = 0)

    # theta_k_diff = np.diff(np.angle(PhaseShift_order_k[:, :]), axis = 0)
    # theta_k_dot_avg = np.mean(theta_k_diff, axis = 0)

    # theta_slips = np.zeros(F_k_avg.shape)

    # for i in range(num_osc):
    #     if i <= k0:
    #         theta_slips[i] = 0.0
    #     else:
    #         theta_slips[i] = theta_k_dot_avg[i] / F_k_avg[i]


    # print()
    # print()
    # print("Theta Slips Check::: Data: {:10f} || -Sqrt(Omega_k - 1): {:10f}".format(theta_slips[k_unlock], -2.0 * np.sqrt(Omega_k[k_unlock]**2 - 1)))



    ## Task 3
    # Adler_order_k = HDFfileData['AdlerScaleOrderParam'][:, :]

    # ## Compute F_k
    # F_k = (np.array(range(0, num_osc), dtype = 'int64') / 2 * amps) * np.absolute(Adler_order_k[:, :])

    # ## Compute the new time scale
    # tau_k = compute_tau_k(F_k, time, num_osc, kmin, 120) + time[0]

    # ## Use new time scale to interpolate \Phi_k
    # f = interp1d(time[:], np.angle(Adler_order_k[:, 12]))

    # ## Compute the interpolation
    # Phi_k_tau_k = f(tau_k)

    # print(np.diff(tau_k[:10]))

    # dPhi_k_dtau_k = np.diff(Phi_k_tau_k) / (tau_k[1] - tau_k[0])


    # plt.figure()
    # plt.plot(dPhi_k_dtau_k, '-')
    # plt.xlabel(r"$\tau_k$")
    # plt.ylabel(r"$\frac{\mathrm{d} \Phi_k}{\mathrm{d}\tau_k}$")
    # plt.savefig(output_dir + "/dPHIdTAU-k_N[{}]_ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N, alpha, beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()








   # Theta Order Parameter
    # m_end   = 8000
    # m_iter  = 50
    # kmin    = k0 + 1
    # num_osc = amps.shape[0]
    # dof     = int(N / 2 - k0)


    
    # for aa in [0.50, 1.00, 1.25, 1.3, 1.4, 1.5, 1.75, 2.00, 2.5]:

    #     results_dir  = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]".format(N, k0, aa, beta, u0)
    #     filename_CLV = "/CLVData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(m_end * m_iter, m_end, m_iter, 10000, int(N / 2 - k0))
    #     SOLVERfilename = "/SolverData_ITERS[{}]_TRANS[{}]".format(iters, trans)
    #     HDFfileData_CLV  = h5py.File(input_dir + results_dir + filename_CLV + '.h5', 'r')
    #     HDFfileData_SOL  = h5py.File(input_dir + results_dir + SOLVERfilename + '.h5', 'r')

    #     amps = HDFfileData_SOL['Amps'][:]
    #     SOL_order = HDFfileData_SOL['ThetaTimeScaleOrderParam'][-1, :]
    #     SOL_R_k = np.absolute(SOL_order)
    #     Adler_order_k = HDFfileData_SOL['AdlerScaleOrderParam'][:, :]

    #     amp_norm = amp_normalization(amps, num_osc, k0)
        
    #     P_k = np.mean( np.sqrt(np.array(range(kmin, num_osc), dtype = 'int64')) * np.absolute(Adler_order_k[:, kmin:]), axis = 0)
    #     P_k = P_k / np.amax(P_k, axis = 0)
        
    #     maxclv = HDFfileData_CLV['MaxCLVStats'][:]

    #     fig = plt.figure(figsize = (16, 9), tight_layout = False)
    #     gs  = GridSpec(1, 1)

    #     ax2 = fig.add_subplot(gs[0, 0])
    #     ax2.plot(range(k0 + 1, num_osc), SOL_R_k[k0 + 1:], '-')
    #     ax2.plot(range(k0 + 1, num_osc), np.absolute(maxclv[:] - np.mean(maxclv[:])) * 10**6, '-')
    #     ax2.plot(range(k0 + 1, num_osc), P_k[:], '-')
    #     ax2.set_title(r"$N = {}, k_0 = {}, \alpha = {:0.3f}$".format(N, k0, aa))
    #     ax2.set_xlabel(r"$k$")
    #     plt.legend([r"$R_k$", r"$e_k$", r"$F_k$"])

    #     plt.savefig(output_dir + "/R_k_vs_CLV_N_[{}]ALPHA[{:.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N, aa, beta, k0, iters, u0), bbox_inches='tight') 
    #     plt.close() 

    #     HDFfileData.close()
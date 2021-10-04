######################
##  Library Imports
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file

import matplotlib.pyplot as plt
# plt.style.use('seaborn-talk')
# mpl.rcParams['figure.figsize']    = [12, 7]
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

from functions import open_file, compute_current_triads, compute_triads_all, read_in_triads



@njit
def normalize_time_order(order, kmin):

    ## Get params
    num_t_steps = order.shape[0]
    num_osc     = order.shape[1]
    
    norm_order = np.ones((num_t_steps, num_osc)) * np.complex(0.0, 0.0)
    for t in range(num_t_steps):
        for k in range(kmin, num_osc):
            norm_order[t, k] = order[t, k] / np.absolute(order[t, k])

    return norm_order


@njit
def compute_time_order(order, kmin):

    ## Get params
    num_t_steps = order.shape[0]
    num_osc     = order.shape[1]

    tmp_time_order = np.ones((num_osc, )) * np.complex(0.0, 0.0)   
    time_order     = np.ones((num_t_steps, num_osc)) * np.complex(0.0, 0.0) 

    t_count = 1
    for t in range(num_t_steps):
        for k in range(kmin, num_osc):
            tmp_time_order[k] += np.exp(1j * np.angle(order[t, k]))
            time_order[t, k] = tmp_time_order[k] / t_count
        t_count += 1

    return time_order


@njit
def orderedtriads_sync_phase(phases, kmin, kmax):
    
    ordered_triads_sync_phase = np.ones(phases.shape) * np.complex(0.0, 0.0)
    triad = np.complex(0.0, 0.0)
    
    for t in range(phases.shape[0]):
        for k in range(kmin, kmax + 1):
            triad = np.complex(0.0, 0.0)
            for k1 in range(-kmax + k, kmax + 1):
                if np.absolute(k1) >= kmin and np.absolute(k - k1) >= kmin:
                        triad += amps[k1] * amps[k - k1] * np.exp(np.complex(0.0, 1.0) * (np.sign(k - k1) * phases[t, np.absolute(k1)] + np.sign(k1) * phases[t, np.absolute(k - k1)] - np.sign(k1 * (k - k1)) * phases[t, k]))
            ordered_triads_sync_phase[t, k] = np.complex(0.0, 1.0) * triad

    return ordered_triads_sync_phase


@njit
def alltriads_sync_phase(phases, kmin, kmax):
    
    ordered_triads_sync_phase = np.ones(phases.shape) * np.complex(0.0, 0.0)
    triad = np.complex(0.0, 0.0)
    
    for t in range(phases.shape[0]):
        for k in range(kmin, kmax + 1):
            triad = np.complex(0.0, 0.0)
            for k1 in range(-kmax + k, kmax + 1):
                if np.absolute(k1) >= kmin and np.absolute(k - k1) >= kmin:
                    if k1 < 0 and k - k1 > 0:
                        triad += amps[k1] * amps[k - k1] * np.exp(np.complex(0.0, 1.0) * ( - phases[t, np.absolute(k1)] + phases[t, k - k1] - phases[t, k]))
                    if k1 > 0 and k - k1 < 0:
                        triad += amps[k1] * amps[k - k1] * np.exp(np.complex(0.0, 1.0) * (phases[t, np.absolute(k1)] - phases[t, np.absolute(k - k1)] - phases[t, k]))
                    elif k1 > 0 and k - k1 > 0:
                        triad += amps[k1] * amps[k - k1] * np.exp(np.complex(0.0, 1.0) * (phases[t, np.absolute(k1)] + phases[t, k - k1] - phases[t, k]))
            ordered_triads_sync_phase[t, k] = np.complex(0.0, 1.0) * triad

    return ordered_triads_sync_phase


@njit
def compute_all_triads_all(phases, kmin, kmax):
    
    triads_non_ordered = np.ones((kmax + 1, 2 * kmax + 1, phases.shape[0])) * (-10.0)

    for t in range(phases.shape[0]):
        for k in range(kmin, kmax + 1):
            for k1 in range(-kmax + k, kmax + 1):
                if np.absolute(k1) >= kmin and np.absolute(k - k1) >= kmin:
                    if k1 < 0 and k - k1 > 0:
                        triads_non_ordered[k, k1 - (-kmax), t] = -phases[t, np.absolute(k1)] + phases[t, np.absolute(k - k1)] - phases[t, k]
                    if k1 > 0 and k - k1 < 0:
                        triads_non_ordered[k, k1 - (-kmax), t] = phases[t, k1] - phases[t, np.absolute(k - k1)] - phases[t, k]
                    elif k1 > 0 and k - k1 > 0:
                        triads_non_ordered[k, k1 - (-kmax), t] = phases[t, k1] + phases[t, k - k1] - phases[t, k]       
                        
    return triads_non_ordered

# @njit
# def compute_all_triads_current(phases, kmin, kmax, t_star):
        
#     triads_non_ordered = np.ones((kmax + 1, 2 * kmax + 1)) * (-10.0)

#     for k in range(kmin, kmax + 1):
#         for k1 in range(-kmax + k, kmax + 1):
#             # if k - k1 <= -kmin:
#             #     if k1 <= -kmin and k1 >= -kmax:
#             #         triads_non_ordered[k, k1 - (-kmax)] = -phases[t_star, np.absolute(k1)] + phases[t_star, np.absolute(k - k1)] - phases[t_star, k]
#             #     if k1 >=  kmin and k1 <= kmax:
#             #         triads_non_ordered[k, k1 - (-kmax)] = phases[t_star, k1] - phases[t_star, np.absolute(k - k1)] - phases[t_star, k]
#             # elif k - k1 >= kmin:
#             #     if k1 <= -kmin and k1 >= -kmax:
#             #         triads_non_ordered[k, k1 - (-kmax)] = -phases[t_star, np.absolute(k1)] + phases[t_star, k - k1] - phases[t_star, k]
#             #     if k1 >=  kmin and k1 <= kmax:
#             #         triads_non_ordered[k, k1 - (-kmax)] = phases[t_star, k1] + phases[t_star, k - k1] - phases[t_star, k] 
                    
#             if np.absolute(k - k1) >= kmin:
#                 triads_non_ordered[k, k1 - (-kmax)] = (phases[t_star, np.absolute(k1)] + phases[t_star, np.absolute(k - k1)] - phases[t_star, k]) * np.sign(k1 * (k - k1))

#     return triads_non_ordered

@njit
def compute_seperated_triads_all(phases, kmin, kmax):
    
    triads_non_ordered = np.ones((kmax + 1, 2 * kmax + 1, phases.shape[0])) * (-10.0)
    triads_ordered     = np.ones((kmax + 1, 2 * kmax + 1, phases.shape[0])) * (-10.0)
    
    for t in range(phases.shape[0]):
        for k in range(kmin, kmax + 1):
            for k1 in range(-kmax + k, kmax + 1):
                if k > k1 and k - k1 >= k1:
                    if k - k1 <= -kmin and k - k1 >= -kmax:
                        if k1 <= -kmin and k1 >= -kmax:
                            triads_ordered[k, k1 - (-kmax), t] = -phases[t, np.absolute(k1)] - phases[t, np.absolute(k - k1)] - phases[t, k]
                        elif k1 >= kmin and k1 <= kmax:
                            triads_ordered[k, k1 - (-kmax), t] = phases[t, k1] - phases[t, np.absolute(k - k1)] - phases[t, k]
                    elif k - k1 >= kmin and k - k1 <= kmax:
                        if k1 <= -kmin and k1 >= -kmax:
                            triads_ordered[k, k1 - (-kmax), t] = -phases[t, np.absolute(k1)] + phases[t, k - k1] - phases[t, k]
                        elif k1 >=  kmin and k1 <= kmax:
                            triads_ordered[k, k1 - (-kmax), t] = phases[t, k1] + phases[t, k - k1] - phases[t, k]
                else:
                    if k - k1 <= -kmin and k - k1 >= -kmax:
                        if k1 <= -kmin and k1 >= -kmax:
                            triads_non_ordered[k, k1 - (-kmax), t] = -phases[t, np.absolute(k1)] - phases[t, np.absolute(k - k1)] - phases[t, k]
                        elif k1 >=  kmin and k1 <= kmax:
                            triads_non_ordered[k, k1 - (-kmax), t] = phases[t, k1] - phases[t, np.absolute(k - k1)] - phases[t, k]
                    elif k - k1 >= kmin and k - k1 <= kmax:
                        if k1 <= -kmin and k1 >= -kmax:
                            triads_non_ordered[k, k1 - (-kmax), t] = -phases[t, np.absolute(k1)] + phases[t, k - k1] - phases[t, k]
                        elif k1 >=  kmin and k1 <= kmax:
                            triads_non_ordered[k, k1 - (-kmax), t] = phases[t, k1] + phases[t, k - k1] - phases[t, k]   
                        
    return triads_non_ordered, triads_ordered


@njit
def heaviside_alltriads_phase(phases, kmin, kmax):
    triad_order = np.ones(phases.shape) * np.complex(0.0, 0.0)
    triad = np.complex(0.0, 0.0)

    for t in range(phases.shape[0]):
        for k in range(kmin, kmax + 1):
            triad = np.complex(0.0, 0.0)
            for k1 in range(-kmax + k, kmax + 1):
                if np.absolute(k1) >= kmin and np.absolute(k - k1) >= kmin:
                    if k1 < 0 and k - k1 > 0:
                        triad += np.exp(np.complex(0.0, 1.0) * (-phases[t, np.absolute(k1)] + phases[t, np.absolute(k - k1)] - phases[t, k]))
                    if k1 > 0 and k - k1 < 0:
                        triad += np.exp(np.complex(0.0, 1.0) * (phases[t, np.absolute(k1)] - phases[t, np.absolute(k - k1)] - phases[t, k]))
                    elif k1 > 0 and k - k1 > 0:
                        triad += np.exp(np.complex(0.0, 1.0) * (phases[t, np.absolute(k1)] + phases[t, k - k1] - phases[t, k]))

            triad_order[t, k] = np.complex(0.0, 1.0) * triad

    return triad_order


@njit
def heaviside_orderedtriads_phase(phases, kmin, kmax):
    triad_order = np.ones(phases.shape) * np.complex(0.0, 0.0)
    triad = np.complex(0.0, 0.0)

    for t in range(phases.shape[0]):
        for k in range(kmin, kmax + 1):
            triad = np.complex(0.0, 0.0)
            for k1 in range(-kmax + k, kmax + 1):
                if np.absolute(k1) >= kmin and np.absolute(k - k1) >= kmin:
                    triad += np.exp(np.complex(0.0, 1.0) * (np.sign(k - k1) * phases[t, np.absolute(k1)] + np.sign(k1) * phases[t, np.absolute(k - k1)] - np.sign(k1 * (k - k1)) * phases[t, k]))

            triad_order[t, k] = np.complex(0.0, 1.0) * triad

    return triad_order


# def pdf(data, nbins)

if __name__ == '__main__':
    ############################
    ##  Get Input Parameters  ##
    ############################
    k0    = 1
    N     = [256, 512, 1024, 2048, 4096, 8192, 16384]
    alpha = np.arange(0.0, 2.51, 0.1) # np.array([0.0, 0.5, 1.0, 1.2, 1.5, 1.7, 2.0, 2.2, 2.5])
    beta  = 0.0
    u0    = 'RANDOM'
    iters = int(4e5)
    trans = int(1e7)

    kmin    = k0 + 1
    num_osc = int(N[0] / 2) + 1
    kmax    = int(N[0] / 2)

    ##########################
    ##  Input & Output Dir  ##
    ##########################
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/PhaseOrder/Investigate"


    ## Open file
    # for i, a in enumerate(alpha):

    #     file = open_file(a, N[0], k0, beta, u0, iters, trans, input_dir)

    #     ## Read in data
    #     time    = file["Time"][:]
    #     amps    = file["Amps"][:]
    #     phases  = file["Phases"][:, :]
    #     R_k_avg = file["R_k_avg"][:]
    #     order   = file["PhaseShiftScaleOrderParam"][:, :]
    #     solver_triads = read_in_triads(file)

    #     ## Close file
    #     file.close()




    #     ##########################
    #     ##  Compute Parameters  ##
    #     ##########################
    #     time_order = compute_time_order(order, k0 + 1)
    #     r_k        = np.absolute(time_order[-1, :])
    #     Theta_k    = np.angle(time_order[-1, :])


    #     # ##########################
    #     # ##    Plot Parameters   ##
    #     # ##########################

    #     # ## Plot Rtilda_k
    #     # fig = plt.figure(figsize = (16, 9), tight_layout = False)
    #     # xg = 4
    #     # yg = 4
    #     # gs  = GridSpec(xg, yg, hspace = 0.3, wspace = 0.3)  

    #     # ax = []
    #     # for i in range(xg):
    #     #     for j in range(yg):
    #     #         ax.append(fig.add_subplot(gs[i, j]))

    #     # count = 0
    #     # plot = 1
    #     # for i, k in enumerate(range(k0 + 1, 100)):    
    #     #     ax[count].plot(np.unwrap(np.angle(order[:, k])))
    #     # #     ax[count].plot(np.mod(np.angle(order[:, k]) + np.pi/2, 2.0 * np.pi) - np.pi/2)
    #     #     ax[count].set_xlabel(r"$t$")
    #     #     ax[count].set_ylabel(r"$\theta_k$")
    #     #     ax[count].legend(["$k = {}$".format(k)])
    #     # #     ax[count].set_ylim(-np.pi/2-0.25, 3.0*np.pi/2 + 0.25)
    #     # #     ax[count].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    #     # #     ax[count].set_yticklabels([r"$-\pi$", r"$- \frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
            
    #     #     if count == 15:
    #     #         plt.savefig(output_dir + "/THETA_K_k[{}]_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(plot, beta, k0, iters, u0), bbox_inches='tight')
    #     #         for i in range(xg * yg):
    #     #             ax[i].cla()
            
    #     #     plot += 1
    #     #     count += 1
    #     #     count = np.mod(count, 16)


    #     # ## Plot theta_k
    #     # fig = plt.figure(figsize = (16, 9), tight_layout = False)
    #     # xg = 4
    #     # yg = 4
    #     # gs  = GridSpec(xg, yg, hspace = 0.3, wspace = 0.3)  

    #     # ax = []
    #     # for i in range(xg):
    #     #     for j in range(yg):
    #     #         ax.append(fig.add_subplot(gs[i, j]))

    #     # count = 0
    #     # plot = 1
    #     # for i, k in enumerate(range(k0 + 1, 18)):    
    #     #     ax[count].plot(np.absolute(order[:, k]))
    #     # #     ax[count].plot(np.absolute(normed_time_order[:, k]), '--')
    #     #     ax[count].set_xlabel(r"$t$")
    #     #     ax[count].set_ylabel(r"$\tilda{R}_k$")
    #     #     ax[count].legend(["$k = {}$".format(k)])
    #     # #     ax[count].set_yscale('log')
    #     # #     ax[count].set_ylim(0, 1)
    #     # #     ax[count].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    #     # #     ax[count].set_yticklabels([r"$-\pi$", r"$- \frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
            
    #     #     if count == 15:
    #     #         plt.savefig(output_dir + "/Rtilda_K_k[{}]_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(plot, beta, k0, iters, u0), bbox_inches='tight')
    #     #         for i in range(xg * yg):
    #     #             ax[i].cla()
            
    #     #     plot += 1
    #     #     count += 1
    #     #     count = np.mod(count, 16)
    #     #     
            






    #     ##################################################################
    #     ##
    #     ##
    #     ##  Is the same effect seen for the ordered tirads?
    #     ##
    #     ##
    #     ##################################################################
    #     # ## Compute all triads - ordered and non ordered
    #     # all_triads = compute_non_odered_triads_all(phases, kmin, kmax)
    #     # print("all")

    #     # ## Compute ordered triads
    #     # ordered_triads = compute_triads_all(phases, kmin, kmax)
    #     # print("ordered")

    #     # ## Plot both
    #     # fig = plt.figure(figsize = (16, 9), tight_layout = False)
    #     # gs  = GridSpec(1, 1, hspace = 0.3, wspace = 0.3)  
    #     # ax  = fig.add_subplot(gs[0, 0])
    #     # counts, bins = np.histogram(all_triads, nbins = 1000, density = True)
    #     # bin_centrs   = (bins[1:] + bins[:-1]) * 0.5
    #     # ax.plot(bin_centrs, counts)
        
    #     # counts, bins = np.histogram(np.extract(ordered_triads != -10.0, ordered_triads), nbins = 1000, density = True)
    #     # bin_centrs   = (bins[1:] + bins[:-1]) * 0.5
    #     # ax.plot(bin_centrs, counts)

    #     # ax.set_xlabel(r"Triads")
    #     # ax.set_ylabel(r"PDF")
    #     # ax.legend([r"All", r"Ordered"])

    #     # plt.savefig(output_dir + "/Hist_Triads_N[{}]_ALPHA[{}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N[0], alpha[0], beta, k0, iters, u0), bbox_inches='tight')
    #     # plt.close()
        


    #     ## Compute the sync phase parameter using only ordered
    #     ordered_order = orderedtriads_sync_phase(phases, kmin, kmax)
    #     print("Ordered")

    #     ## Compute the time order parameter for the ordered triads
    #     ordered_time_order = compute_time_order(ordered_order, kmin)
    #     print("Time Order")


    #     # ## Compute the non ordered triads sync phase by subtracting the two
    #     # non_ordered_order = order - ordered_order
    #     # print("Non Ordered")

    #     # ## Compute time order parameter for this
    #     # non_ordered_time_order = compute_time_order(non_ordered_order, kmin)
    #     # print("Non Ordered Time")





    #     ## Compare R_k and Theta_k parameters 
    #     fig = plt.figure(figsize = (32, 16), tight_layout = False)
    #     xg = 2
    #     yg = 2
    #     gs = GridSpec(xg, yg, hspace = 0.3, wspace = 0.3)  

    #     ax = []
    #     for i in range(xg):
    #         for j in range(yg):
    #             ax.append(fig.add_subplot(gs[i, j]))

    #     ax[0].plot(r_k)
    #     ax[0].set_xlabel(r"$k$")
    #     ax[0].set_ylabel(r"$R_k$")
    #     ax[0].legend(["All Triads"])

    #     ax[1].plot(Theta_k, '.')
    #     ax[1].set_xlabel(r"$k$")
    #     ax[1].set_ylabel(r"$\Theta_k$")
    #     ax[1].legend(["All Triads"])

    #     ax[2].plot(np.absolute(ordered_time_order[-1, :]))
    #     ax[2].set_xlabel(r"$k$")
    #     ax[2].set_ylabel(r"$R_k$")
    #     ax[2].legend(["Ordered Triads"])

    #     ax[3].plot(np.angle(ordered_time_order[-1, :]), '.')
    #     ax[3].set_xlabel(r"$k$")
    #     ax[3].set_ylabel(r"$\Theta_k$")
    #     ax[3].legend(["Ordered Triads"])

    #     # ax[4].plot(np.absolute(non_ordered_time_order[-1, :]))
    #     # ax[4].set_xlabel(r"$k$")
    #     # ax[4].set_ylabel(r"$R_k$")
    #     # ax[4].legend(["Non Ordered Triads"])

    #     # ax[5].plot(np.angle(non_ordered_time_order[-1, :]), '.')
    #     # ax[5].set_xlabel(r"$k$")
    #     # ax[5].set_ylabel(r"$\Theta_k$")
    #     # ax[5].legend(["Non Ordered Triads"])

    #     plt.savefig(output_dir + "/All_Ordered_NonOrdered_Compare_Triads_N[{}]_ALPHA[{}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N[0], a, beta, k0, iters, u0), bbox_inches='tight')
    #     plt.close()


    #     fig = plt.figure(figsize = (32, 16), tight_layout = False)
    #     xg = 1
    #     yg = 2
    #     gs = GridSpec(xg, yg, hspace = 0.3, wspace = 0.3)  

    #     ax = []
    #     for i in range(xg):
    #         for j in range(yg):
    #             ax.append(fig.add_subplot(gs[i, j]))

    #     ax[0].plot(r_k)
    #     ax[0].plot(np.absolute(ordered_time_order[-1, :]))
    #     # ax[0].plot(np.absolute(non_ordered_time_order[-1, :]))
    #     ax[0].set_xlabel(r"$k$")
    #     ax[0].set_ylabel(r"$R_k$")
    #     ax[0].legend([r"All", r"Ordered", r"Non Ordered"])

    #     counts, bins = np.histogram(Theta_k, bins = 200, density = True)
    #     bin_centrs   = (bins[1:] + bins[:-1]) * 0.5
    #     ax[1].plot(bin_centrs, counts)
    #     counts, bins = np.histogram(np.angle(ordered_time_order[-1, :]), bins = 200, density = True)
    #     bin_centrs   = (bins[1:] + bins[:-1]) * 0.5
    #     ax[1].plot(bin_centrs, counts)
    #     # counts, bins = np.histogram(np.angle(non_ordered_time_order[-1, :]), bins = 200, density = True)
    #     # bin_centrs   = (bins[1:] + bins[:-1]) * 0.5
    #     ax[1].plot(bin_centrs, counts)
    #     ax[1].set_xlabel(r"$k$")
    #     ax[1].set_ylabel(r"$\Theta_k$")
    #     ax[1].legend([r"All", r"Ordered", r"Non Ordered"])

    #     plt.savefig(output_dir + "/Compare_Triads_N[{}]_ALPHA[{}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N[0], a, beta, k0, iters, u0), bbox_inches='tight')
    #     plt.close()






    #     ## Compute the heavside version of the sync phase
    #     heaviside_all_order = heaviside_alltriads_phase(phases, kmin, kmax)

    #     ## Compute the time order parameter for the ordered triads
    #     heaviside_all_time_order = compute_time_order(heaviside_all_order, kmin)
    #     # print("Time Order")


    #     ## Compute the heaviside version using ordered triads
    #     heaviside_ordered_order = heaviside_orderedtriads_phase(phases, kmin, kmax)

    #     ## Compute the time order parameter for the ordered triads
    #     heaviside_ordered_time_order = compute_time_order(heaviside_ordered_order, kmin)
    #     # print("Time Order")

    #     # ## Compute the non ordered triads sync phase by subtracting the two
    #     # heaviside_non_ordered_order = heaviside_all_order - heaviside_ordered_order
    #     # print("Non Ordered")

    #     # ## Compute time order parameter for this
    #     # non_ordered_time_order = compute_time_order(heaviside_non_ordered_order, kmin)
    #     # print("Non Ordered Time")


    #     fig = plt.figure(figsize = (16, 9), tight_layout = False)
    #     xg = 1
    #     yg = 1
    #     gs = GridSpec(xg, yg, hspace = 0.3, wspace = 0.3)  

    #     ax = []
    #     for i in range(xg):
    #         for j in range(yg):
    #             ax.append(fig.add_subplot(gs[i, j]))

    #     ax[0].plot(r_k)
    #     ax[0].plot(np.absolute(ordered_time_order[-1, :]), '.-')
    #     ax[0].plot(np.absolute(heaviside_all_time_order[-1, :]))
    #     ax[0].plot(np.absolute(heaviside_ordered_time_order[-1, :]))
    #     ax[0].set_xlabel(r"$k$")
    #     ax[0].set_ylabel(r"$R_k$")
    #     ax[0].legend([r"All", r"Ordered", r"Heaviside", r"Heaviside Ordered"])

    #     # counts, bins = np.histogram(Theta_k, bins = 200, density = True)
    #     # bin_centrs   = (bins[1:] + bins[:-1]) * 0.5
    #     # ax[1].plot(bin_centrs, counts)
    #     # counts, bins = np.histogram(np.angle(ordered_time_order[-1, :]), bins = 200, density = True)
    #     # bin_centrs   = (bins[1:] + bins[:-1]) * 0.5
    #     # ax[1].plot(bin_centrs, counts)
    #     # counts, bins = np.histogram(np.angle(heaviside_all_time_order[-1, :]), bins = 200, density = True)
    #     # bin_centrs   = (bins[1:] + bins[:-1]) * 0.5
    #     # ax[1].plot(bin_centrs, counts)
    #     # counts, bins = np.histogram(np.angle(heaviside_ordered_time_order[-1, :]), bins = 200, density = True)
    #     # bin_centrs   = (bins[1:] + bins[:-1]) * 0.5
    #     # ax[1].plot(bin_centrs, counts)
    #     # ax[1].set_xlabel(r"$k$")
    #     # ax[1].set_ylabel(r"$\Theta_k$")
    #     # ax[1].legend([r"All", r"Ordered", r"Heaviside", r"Heaviside Ordered"])

    #     plt.savefig(output_dir + "/Compare_Heaviside_N[{}]_ALPHA[{}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(N[0], a, beta, k0, iters, u0), bbox_inches='tight')
    #     plt.close()







    rk_sum          = np.zeros((len(N), len(alpha)))
    order_sum       = np.zeros((len(N), len(alpha)))
    heavi_sum       = np.zeros((len(N), len(alpha)))
    heavi_order_sum = np.zeros((len(N), len(alpha))) 


    for i, n in enumerate(N):

        R_k_avg          = np.zeros((len(alpha), int(n / 2 + 1)))
        Heaviside        = np.zeros((len(alpha), int(n / 2 + 1)))
        Ordered          = np.zeros((len(alpha), int(n / 2 + 1)))
        HeavisideOrdered = np.zeros((len(alpha), int(n / 2 + 1)))

        for j, a in enumerate(alpha):

            file = open_file(a, n, k0, beta, u0, iters, trans, input_dir)

            ## Read in data
            # time    = file["Time"][:]
            # amps    = file["Amps"][:]
            # phases  = file["Phases"][:, :]
            # order   = file["PhaseShiftScaleOrderParam"][:, :]
            # solver_triads = read_in_triads(file)
            R_k_avg[j, :]          = file["R_k_avg"][:]
            Heaviside[j, :]        = file["HeavisideSyncPhase"][:]
            Ordered[j, :]          = file["OrderedSyncPhase"][:]
            HeavisideOrdered[j, :] = file["HeavisideOrderedSyncPhase"][:]

            ## Close file
            file.close()

            ## Average phase sync
            rk_sum[i, j]          = np.mean(R_k_avg[j, k0 + 1:])
            order_sum[i, j]       = np.mean(Heaviside[j, k0 + 1:])
            heavi_sum[i, j]       = np.mean(Ordered[j, k0 + 1:])
            heavi_order_sum[i, j] = np.mean(HeavisideOrdered[j, k0 + 1:])




    ## Plot all four average phase sync parameters
    fig = plt.figure(figsize = (16, 9), tight_layout = False)
    xg = 2
    yg = 2
    gs = GridSpec(xg, yg, hspace = 0.3, wspace = 0.3)  

    ax = []
    for i in range(xg):
        for j in range(yg):
            ax.append(fig.add_subplot(gs[i, j]))

    for i in range(len(N)):
        ax[0].plot(alpha, rk_sum[i, :])
        ax[1].plot(alpha, order_sum[i, :])
        ax[2].plot(alpha, heavi_sum[i, :])
        ax[3].plot(alpha, heavi_order_sum[i, :])
    for i in range(xg * yg):
        ax[i].set_xlabel(r"$\alpha$")
        ax[i].set_xlim(alpha[0], alpha[-1])
        ax[i].set_ylim(0, 1.1)
        ax[i].legend([r"$N = {}$".format(n) for n in N])
    ax[0].set_title(r"$R_k$")
    ax[1].set_title(r"Ordered")
    ax[2].set_title(r"Heaviside")
    ax[3].set_title(r"Heaviside & Ordered")


    plt.savefig(output_dir + "/Compare_different_defs_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(beta, k0, iters, u0), bbox_inches='tight')
    plt.close()




    ## Create figures
    fig = plt.figure(figsize = (16, 9), tight_layout = False)
    xg = 3
    yg = 3
    gs = GridSpec(xg, yg, hspace = 0.3, wspace = 0.3)  

    ax = []
    for i in range(xg):
        for j in range(yg):
            ax.append(fig.add_subplot(gs[i, j]))

    plotting_alpha = np.array([0.0, 0.5, 1.0, 1.2, 1.5, 1.7, 2.0, 2.2, 2.5])
    


    for i, n in enumerate(N):

        krange = range(k0 + 1, int(n / 2 + 1))

        for j, a in enumerate(plotting_alpha):

            file = open_file(a, n, k0, beta, u0, iters, trans, input_dir)

            R_k_avg          = file["R_k_avg"][:]
            # Heaviside        = file["HeavisideSyncPhase"][:]
            # Ordered          = file["OrderedSyncPhase"][:]
            # HeavisideOrdered = file["HeavisideOrderedSyncPhase"][:]

            ## Close file
            file.close()

            ## Plot
            ax[j].plot(krange, R_k_avg[k0 + 1:])
            ax[j].set_xscale('log')
            # ax[j].set_xlabel(r"$k$")
            # ax[j].set_ylabel(r"$R_k$")

    plt.savefig(output_dir + "/R_k_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(beta, k0, iters, u0), bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize = (16, 9), tight_layout = False)
    xg = 3
    yg = 3
    gs = GridSpec(xg, yg, hspace = 0.3, wspace = 0.3)  

    ax = []
    for i in range(xg):
        for j in range(yg):
            ax.append(fig.add_subplot(gs[i, j]))


    for i, n in enumerate(N):

        krange = range(k0 + 1, int(n / 2 + 1))

        for j, a in enumerate(plotting_alpha):

            file = open_file(a, n, k0, beta, u0, iters, trans, input_dir)

            # R_k_avg          = file["R_k_avg"][:]
            Heaviside        = file["HeavisideSyncPhase"][:]
            # Ordered          = file["OrderedSyncPhase"][:]
            # HeavisideOrdered = file["HeavisideOrderedSyncPhase"][:]

            ## Close file
            file.close()

            ## Plot
            ax[j].plot(krange, Heaviside[k0 + 1:])
            ax[j].set_xscale('log')
            # ax[j].set_xlabel(r"$k$")
            # ax[j].set_ylabel(r"$R_k$")
    
    plt.suptitle(r"Heaviside")
    plt.savefig(output_dir + "/R_k_Heaviside_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(beta, k0, iters, u0), bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize = (16, 9), tight_layout = False)
    xg = 3
    yg = 3
    gs = GridSpec(xg, yg, hspace = 0.3, wspace = 0.3)  

    ax = []
    for i in range(xg):
        for j in range(yg):
            ax.append(fig.add_subplot(gs[i, j]))

    for i, n in enumerate(N):

        krange = range(k0 + 1, int(n / 2 + 1))

        for j, a in enumerate(plotting_alpha):

            file = open_file(a, n, k0, beta, u0, iters, trans, input_dir)

            # R_k_avg          = file["R_k_avg"][:]
            # Heaviside        = file["HeavisideSyncPhase"][:]
            Ordered          = file["OrderedSyncPhase"][:]
            # HeavisideOrdered = file["HeavisideOrderedSyncPhase"][:]

            ## Close file
            file.close()

            ## Plot
            ax[j].plot(krange, Ordered[k0 + 1:])
            ax[j].set_xscale('log')
            # ax[j].set_xlabel(r"$k$")
            # ax[j].set_ylabel(r"$R_k$")

    plt.suptitle(r"Ordered")
    plt.savefig(output_dir + "/R_k_Ordered_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(beta, k0, iters, u0), bbox_inches='tight')
    plt.close()


    fig = plt.figure(figsize = (16, 9), tight_layout = False)
    xg = 3
    yg = 3
    gs = GridSpec(xg, yg, hspace = 0.3, wspace = 0.3)  

    ax = []
    for i in range(xg):
        for j in range(yg):
            ax.append(fig.add_subplot(gs[i, j]))


    for i, n in enumerate(N):

        krange = range(k0 + 1, int(n / 2 + 1))

        for j, a in enumerate(plotting_alpha):

            file = open_file(a, n, k0, beta, u0, iters, trans, input_dir)

            # R_k_avg          = file["R_k_avg"][:]
            # Heaviside        = file["HeavisideSyncPhase"][:]
            # Ordered          = file["OrderedSyncPhase"][:]
            HeavisideOrdered = file["HeavisideOrderedSyncPhase"][:]

            ## Close file
            file.close()

            ## Plot
            ax[j].plot(krange, HeavisideOrdered[k0 + 1:])
            ax[j].set_xscale('log')
            # ax[j].set_xlabel(r"$k$")
            # ax[j].set_ylabel(r"$R_k$")

    plt.suptitle(r"HeavisideOrdered")
    plt.savefig(output_dir + "/R_k_HeavisideOrdered_N[VARIED]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.png".format(beta, k0, iters, u0), bbox_inches='tight')
    plt.close()
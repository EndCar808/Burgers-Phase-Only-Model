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
plt.style.use('seaborn-talk')
mpl.rcParams['figure.figsize'] = [10, 8]
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['axes.labelsize'] = 10
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import h5py
import sys
import os
import time as TIME
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mprocs
from itertools import zip_longest
from subprocess import Popen, PIPE
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from functions import compute_time
from numba import njit


import pyfftw as fftw

##########################
##    FUNCTION DEFS     ##
##########################
@njit
def compute_realspace_rms(u):

    ## Allocate arrays
    u_rms       = np.zeros((u.shape[0], ))
    u_rms_field = np.zeros(u.shape)

    ## Compute rms
    for i in range(u.shape[0]):
        u_rms[i]       = np.sqrt(np.mean(u[i, :]**2))
        u_rms_field[i] = u[i, :] / u_rms[i]


    return u_rms, u_rms_field

@njit
def compute_grad_rms(du_x):

    ## Allocate arrays
    du_x_rms       = np.zeros((u.shape[0], ))
    du_x_rms_field = np.zeros(u.shape)

    for i in range(du_x.shape[0]):
        du_x_rms[i]          = np.sqrt(np.mean(du_x[i, :]**2))
        du_x_rms_field[i, :] = du_x[i, :] / du_x_rms[i]

    return


def plot_spacetime_inspection(u_rms, du_x_rms, N, alpha, beta, iters, k0, t):

    ## Create figure
    fig = plt.figure(figsize = (16, 12), tight_layout=True)
    gs  = GridSpec(4, 2)

    t_shock = u.shape[0] - 800 - t
    t_chaos = u.shape[0] - 100 - t

    ## REAL SPACE - SPACETIME
    ax1 = fig.add_subplot(gs[0:2, 0])
    im1 = ax1.imshow(np.flip(u_rms), cmap = "bwr") 
    ax1.axhline(y = t_shock, xmin = 0, xmax = N, color = 'blue')
    ax1.axhline(y = t_chaos, xmin = 0, xmax = N, color = 'black')
    ax1.set_aspect('auto')
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$t$")
    ax1.set_xticks([0.0, np.ceil(N / 4), np.ceil(2 * N / 4), np.ceil(3 * N / 4), N])
    ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_yticks([0.0, np.ceil(u_rms.shape[0] / 4), np.ceil(2 * u_rms.shape[0] / 4), np.ceil(3 * u_rms.shape[0] / 4), u_rms.shape[0]])
    ax1.set_yticklabels([r"$100000$", r"$7500$", r"$5000$", r"$2500$", r"$0$"])
    # plt.gcf().text(x = 0.001, y = 0.97, s = r"a)", fontsize = 20)
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"$u / u_{rms}$")

    ## REAL SPACE - SPACETIME
    ax1 = fig.add_subplot(gs[0:2, 1])
    im1 = ax1.imshow(np.flip(u_rms[t:1000 + t, :]), cmap = "bwr") 
    ax1.axhline(y = u_rms[t:1000 + t, :].shape[0] - 800, xmin = 0, xmax = N, color = 'blue')
    ax1.axhline(y = u_rms[t:1000 + t, :].shape[0] - 100, xmin = 0, xmax = N, color = 'black')
    ax1.set_aspect('auto')
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$t$")
    ax1.set_xticks([0.0, np.ceil(N / 4), np.ceil(2 * N / 4), np.ceil(3 * N / 4), N])
    ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_yticks([0.0, np.ceil(u_rms[t:1000 + t, :].shape[0] / 4), np.ceil(2 * u_rms[t:1000 + t, :].shape[0] / 4), np.ceil(3 * u_rms[t:1000 + t, :].shape[0] / 4), u_rms[t:1000 + t, :].shape[0]])
    ax1.set_yticklabels([r"$1000$", r"$750$", r"$500$", r"$250$", r"$0$"])
    # plt.gcf().text(x = 0.001, y = 0.97, s = r"a)", fontsize = 20)
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"$u / u_{rms}$")



    ax3 = fig.add_subplot(gs[2, 0:2])
    ax33 = ax3.twinx()
    ax33.set_ylim(-2.9, 2.9)
    ax33.plot(u_rms[t_shock, :], color = 'blue')
    ax3.set_ylabel(r"$u / u_{rms}$")
    ax3.plot(du_x_rms[t_shock, :], color = 'red')
    ax33.set_ylabel(r"$\partial_x u / \partial_x u_{rms}$", color = 'red')
    ax3.set_ylim(-3.5, 3.5)
    ax3.set_xlim(0, N)
    ax3.set_xticks([0.0, np.ceil(N / 4), np.ceil(2 * N / 4), np.ceil(3 * N / 4), N])
    ax3.set_xticklabels([])
    leg1 = mpl.patches.Rectangle((0, 0), 0, 0, alpha = 0.0)
    ax3.legend([leg1], [r"$t=({:04.3f})$".format(t_shock)], handlelength = -0.5, fancybox = True, prop = {'size': 10}, loc = 'upper left')
    # plt.gcf().text(x = 0.001, y = 0.5, s = r"c)", fontsize = 20)


    ax4 = fig.add_subplot(gs[3, 0:2])
    ax44 = ax4.twinx()
    ax44.set_ylim(-3.5, 3.5)
    ax4.set_ylabel(r"$u / u_{rms}$")
    ax44.plot(u_rms[t_chaos, :], color = 'black')
    ax4.set_ylim(-2.9, 2.9)
    ax4.set_xlim(0, N)
    ax4.set_xticks([0.0, np.ceil(N / 4), np.ceil(2 * N / 4), np.ceil(3 * N / 4), N])
    ax4.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax4.set_xlabel(r"$x$")
    ax4.plot(du_x_rms[t_chaos, :], color = 'red')
    ax44.set_ylabel(r"$\partial_x u / \partial_x u_{rms}$", color = 'red')
    ax4.legend([leg1], [r"$t=({:04.3f})$".format(t_chaos)], handlelength = -0.5, fancybox = True, prop = {'size': 10}, loc = 'upper left')
    
    # plt.gcf().text(x = 0.001, y = 0.25, s = r"d)", fontsize = 20)

    plt.savefig(output_dir + "/SNAPS/SPACETIME_SNAP_{:05d}.png".format(t), format='png', dpi = 400)  
    plt.close()



def plot_phase_inspection(phases):

    ## Create figure
    fig = plt.figure(figsize = (16, 9), tight_layout=True)
    gs  = GridSpec(1, 1)

    ax2 = fig.add_subplot(gs[0, 0])
    im2  = ax2.imshow(np.flipud(np.mod(phases, 2.0*np.pi)), cmap = "Blues", vmin = 0.0, vmax = 2.0 * np .pi)
    # ax2.axhline(y = t_shock, xmin = 0, xmax = N, color = 'blue')
    # ax2.axhline(y = t_chaos, xmin = 0, xmax = N, color = 'black')
    ax2.set_aspect('auto')
    ax2.set_xlabel(r"$k$")
    ax2.set_ylabel(r"$t$")
    ax2.set_yticks([0.0, np.ceil(phases.shape[0] / 4), np.ceil(2 * phases.shape[0] / 4), np.ceil(3 * phases.shape[0] / 4), phases.shape[0]])
    ax2.set_yticklabels([r"$1000$", r"$750$", r"$500$", r"$250$", r"$0$"])
    plt.gcf().text(x = 0.49, y = 0.97, s = r"b)", fontsize = 20)
    div2  = make_axes_locatable(ax2)
    cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
    cb2   = plt.colorbar(im2, cax = cbax2)
    cb2.set_ticks([ 0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
    cb2.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
    cb2.set_label(r"$\phi_k$")

    plt.savefig(output_dir + "/Phases_SNAP_{:05d}.png".format(t), format='png', dpi = 400)  
    plt.close()



def compute_phases(u, N):

    ## Create input and ouput fft arrays
    real = fftw.zeros_aligned(N, dtype = 'float64')
    cplx = fftw.zeros_aligned(int(N /2 + 1), dtype = 'complex128')

    ## Set up the fft scheme
    fft_r2c = fftw.builders.rfft(real)
    fft_c2r = fftw.builders.irfft(cplx)

    ## Allocate array for phases
    phases = np.zeros((u.shape[0], int(N /2 + 1)))
    for t in range(u.shape[0]):
        u_z = fft_r2c(u[t, :])
        phases[t, :] = np.angle(u_z) 

    return phases




def plot_spacetime_full(u_urms, du_x_rms, time, phases, N, alpha, beta, k0, t):
    ## Create figure
    fig = plt.figure(figsize = (7.0, 7.0), tight_layout=False)
    gs  = GridSpec(4, 2, hspace = 0.7, wspace = 0.6)

    ## Selct time slice
    t_shock = 700
    t_chaos = 100

    labelsize = 8
    ticksize = 8
    font_size = 10

    u_urms_plot = np.flipud(u_urms)
    phases_plot = np.flipud(np.mod(phases, 2.0*np.pi))
    du_x_rms_plot = np.flipud(du_x_rms)

    ## REAL SPACE - SPACETIME
    ax1 = fig.add_subplot(gs[0:2, 0])
    im1 = ax1.imshow(u_urms_plot, cmap = "bwr") 
    ax1.axhline(y = t_shock, xmin = 0, xmax = N, color = 'blue')
    ax1.axhline(y = t_chaos, xmin = 0, xmax = N, color = 'black')
    ax1.set_aspect('auto')
    ax1.set_xlabel(r"$x$", labelpad = 0.4)
    ax1.set_ylabel(r"$t$")
    ax1.set_xticks([0.0, np.ceil(N / 4), np.ceil(2 * N / 4), np.ceil(3 * N / 4), N])
    ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"], fontsize = font_size)
    ax1.set_yticks([0.0, np.ceil(u_urms.shape[0] / 4), np.ceil(2 * u_urms.shape[0] / 4), np.ceil(3 * u_urms.shape[0] / 4), u_urms.shape[0]])
    ax1.set_yticklabels([r"$1000$", r"$750$", r"$500$", r"$250$", r"$0$"], fontsize = font_size)
    plt.gcf().text(x = 0.015, y = 0.89, s = r"a)", fontsize = font_size)
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"$u / u_{rms}$")
    cb1.ax.tick_params(axis = 'y', labelsize = font_size)

    ## PHASES - SPACETIME
    ax2 = fig.add_subplot(gs[0:2, 1])
    im2  = ax2.imshow(phases_plot, cmap = "Blues", vmin = 0.0, vmax = 2.0 * np .pi)
    ax2.axhline(y = t_shock, xmin = 0, xmax = N, color = 'blue')
    ax2.axhline(y = t_chaos, xmin = 0, xmax = N, color = 'black')
    ax2.set_aspect('auto')
    ax2.set_xlabel(r"$k$", labelpad = 0.8)
    ax2.set_ylabel(r"$t$")
    ax2.tick_params(axis = 'x', labelsize = font_size)
    ax2.set_yticks([0.0, np.ceil(phases.shape[0] / 4), np.ceil(2 * phases.shape[0] / 4), np.ceil(3 * phases.shape[0] / 4), phases.shape[0]])
    ax2.set_yticklabels([r"$1000$", r"$750$", r"$500$", r"$250$", r"$0$"], fontsize = font_size)
    plt.gcf().text(x = 0.49, y = 0.89, s = r"b)", fontsize = font_size)
    div2  = make_axes_locatable(ax2)
    cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
    cb2   = plt.colorbar(im2, cax = cbax2)
    cb2.set_ticks([ 0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
    cb2.ax.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"], fontsize = font_size)
    cb2.set_label(r"$\phi_k$")

    ax3 = fig.add_subplot(gs[2, 0:2])
    ax33 = ax3.twinx()
    ax33.set_ylim(-2.9, 2.9)
    ax33.plot(u_urms_plot[t_shock, :], color = 'blue')
    ax3.set_ylabel(r"$u / u_{rms}$", color = 'blue')
    ax3.plot(du_x_rms_plot[t_shock, :], color = 'red')
    ax33.set_ylabel(r"$\partial_x u / \partial_x u_{rms}$", color = 'red')
    ax3.set_ylim(-2.9, 2.9)
    ax3.set_xlim(0, N)
    ax3.set_xticks([0.0, np.ceil(N / 4), np.ceil(2 * N / 4), np.ceil(3 * N / 4), N])
    ax3.set_xticklabels([])
    ax3.tick_params(axis = 'y', labelsize = font_size)
    ax33.tick_params(axis = 'y', labelsize = font_size)
    plt.gcf().text(x = 0.015, y = 0.45, s = r"c)", fontsize = font_size)
    
    ax4 = fig.add_subplot(gs[3, 0:2])
    ax44 = ax4.twinx()
    ax44.set_ylim(-2.9, 2.9)
    ax4.set_ylabel(r"$u / u_{rms}$")
    ax44.plot(u_urms_plot[t_chaos, :], color = 'black')
    ax4.set_ylim(-2.9, 2.9)
    ax4.set_xlim(0, N)
    ax4.set_xticks([0.0, np.ceil(N / 4), np.ceil(2 * N / 4), np.ceil(3 * N / 4), N])
    ax4.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"], fontsize = font_size)
    ax4.set_xlabel(r"$x$")
    ax44.tick_params(axis = 'y', labelsize = font_size)
    ax4.tick_params(axis = 'y', labelsize = font_size)
    ax4.plot(du_x_rms_plot[t_chaos, :], color = 'red')
    ax44.set_ylabel(r"$\partial_x u / \partial_x u_{rms}$", color = 'red')
    plt.gcf().text(x = 0.015, y = 0.23, s = r"d)", fontsize = font_size)
    
    plt.savefig(output_dir + "/SPACETIME_FULL_t[{}]_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].pdf".format(t, N, alpha, beta, k0, iters), bbox_inches='tight') #, format='png', dpi = 400)  
    plt.close()




######################
##       MAIN       ##
######################
if __name__ == '__main__':
    

    ######################
    ##  Get Input Parameters
    ######################
    if (len(sys.argv) != 8):
        print("No Input Provided, Error.\nProvide: \nk0\nAlpha\nBeta\nIterations\nTransient Iterations\nN\n")
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
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Transfer_Report" 





    ######################
    ##  Read in Input File
    ######################    
    HDFfileData = h5py.File(input_dir + results_dir + filename + '.h5', 'r')

    # print input file name to screen
    print("\n\nData File: {}.h5\n".format(results_dir + filename))


    ######################
    ##  Read in Datasets
    ######################
    u    = HDFfileData['RealSpace'][:, :]
    du_x = HDFfileData['RealSpaceGrad'][:]
    try:
        time   = HDFfileData['Time'][:]
    except:
        time, step = compute_time(N, alpha, beta, k0, iters, trans)

    ######################
    ##  Preliminary Calcs
    ######################
    ntsteps = u.shape[0]
    num_osc = u.shape[1] / 2 + 1
    kmin    = k0 + 1
    kmax    = num_osc - 1


    ######################
    ##  Compute fields
    ######################
    u_rms = u / np.sqrt(np.mean(u[10, :]**2))

    du_x_rms = du_x / np.sqrt(np.mean(du_x[10, :]**2))

    ######################
    ##  Plot
    ######################

    # for t in range(u_rms.shape[0] - 1000):
    #     print("Plotting snap {}".format(t))
    #     plot_spacetime_inspection(u_rms, du_x_rms, N, alpha, beta, iters, k0, t)
    #     
    
    phases = compute_phases(u, N)
    # print(phases.shape)
    # i = 0
    # for t in range(0, u.shape[0], int(u.shape[0] / 100)):
    #     print(i)
    #     plot_phase_inspection(phases[t:(1000 + t), :])
    #     i += 1
        


    # i = 0
    # for t in range(0, u.shape[0], int(u.shape[0] / 100)):
    #     print(i)
    #     # Plot full space-time figure
    #     plot_spacetime_full(u_rms[t:1000 + t, :], du_x_rms[t:1000 + t, :], time[t:1000 + t], phases[t:1000 + t, :], N, alpha, beta, k0, t)
    #     i += 1

    t = 90000
    plot_spacetime_full(u_rms[t:1000 + t, :], du_x_rms[t:1000 + t, :], time[t:1000 + t], phases[t:1000 + t, :], N, alpha, beta, k0, t)
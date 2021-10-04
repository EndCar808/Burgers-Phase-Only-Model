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
plt.style.use('seaborn-talk')
mpl.rcParams['figure.figsize'] = [10, 8]
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
mpl.rcParams['lines.linewidth'] = 1.25
mpl.rcParams['lines.markersize'] = 6
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



######################
##	Real Space Function Defs
######################
def compute_phases_rms(phases):
	phases_rms  = np.sqrt(np.mean(phases[:, :]**2, axis = 1))
	phases_prms = np.array([phases[i, :] / phases_rms[i] for i in range(phases.shape[0])])

	return phases_prms


## Real Space Data
def compute_realspace(amps, phases, N):
	print("\n...Creating Real Space Soln...\n")

	# Create full set of amps and phases
	amps_full   = np.append(amps[:], np.flipud(amps[1:-1]))
	phases_full = np.concatenate((phases[:, :], -np.fliplr(phases[:, 1:-1])), axis = 1)

	# Construct modes and realspace soln
	u_z = amps_full * np.exp(np.complex(0.0, 1.0) * phases_full)
	u   = np.real(np.fft.ifft(u_z, axis = 1, norm = 'ortho'))

	# Compute normalized realspace soln
	u_rms  = np.sqrt(np.mean(u[:, :]**2, axis = 1))
	u_urms = np.array([u[i, :] / u_rms[i] for i in range(u.shape[0])])

	x = np.arange(0, 2*np.pi, 2*np.pi/N)
	
	return u, u_urms, x, u_z

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

def compute_gradient(u_z, kmin, kmax):
	print("\nCreating Gradient\n")
	k            = np.concatenate((np.zeros((kmin)), np.arange(kmin, kmax + 1), -np.flip(np.arange(kmin, kmax)), np.zeros((kmin - 1))))
	grad_u_z     = np.complex(0.0, 1.0) * k * u_z
	du_x         = np.real(np.fft.ifft(grad_u_z, axis = 1, norm = 'ortho'))
	du_x_rms_tmp = np.sqrt(np.mean(du_x ** 2, axis = 1))
	du_x_rms     = np.array([du_x[i, :] / du_x_rms_tmp[i] for i in range(u_z.shape[0])])

	return du_x, du_x_rms



######################
##	Plotting Function Defs
######################
def plot_spacetime(x, u_urms, time, phases, N, alpha, beta, k0):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
	fig.suptitle(r'$N = {} \quad \alpha = {} \quad \beta = {} \quad k_0 = {}$'.format(N, alpha, beta, k0))

	## REAL SPACE
	im1 = ax1.imshow(u_urms, cmap = "bwr")
	ax1.set_aspect('auto')
	ax1.set_title(r"Real Space")
	ax1.set_xlabel(r"$x$")
	ax1.set_ylabel(r"$t$")
	ax1.set_xticks([0.0, np.ceil(N / 4), np.ceil(2 * N / 4), np.ceil(3 * N / 4), N])
	ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	div1  = make_axes_locatable(ax1)
	cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
	cb1   = plt.colorbar(im1, cax = cbax1)
	cb1.set_label(r"$u(x, t) / u^{rms}(x, t)$")

	## PHASES
	im2  = ax2.imshow(np.mod(phases, 2.0*np.pi), cmap = "bwr")
	ax2.set_aspect('auto')
	ax2.set_title(r"Phases")
	ax2.set_xlabel(r"$k$")
	ax2.set_ylabel(r"$t$")
	div2  = make_axes_locatable(ax2)
	cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
	cb2   = plt.colorbar(im2, cax = cbax2)
	cb2.set_ticks([ 0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
	cb2.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
	cb2.set_label(r"$\phi_k(t)$")

	plt.savefig(output_dir + "/SPACETIME_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(N, alpha, beta, k0, iters), format='png', dpi = 400)  
	plt.close()


def plot_grad_realspace(x, u_urms, du_x_rms, N, alpha, beta, k0):
	fig, (ax1, ax3) = plt.subplots(1, 2, figsize = (16, 9))
	fig.suptitle(r'$N = {} \quad \alpha = {} \quad \beta = {} \quad k_0 = {}$'.format(N, alpha, beta, k0))

	## INITIAL CONDITION
	t = 0
	ax1.plot(x, u_urms[t, :], color = 'blue')
	ax1.set_xlim(0, 2.0 * np.pi)
	ax1.set_xlabel(r"$x$")
	leg1 = mpl.patches.Rectangle((0, 0), 0, 0, alpha = 0.0)
	ax1.legend([leg1], [r"$Iter:({})$".format(t)], handlelength = -0.5, fancybox = True, prop = {'size': 10})
	ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, 2*np.pi])
	ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	ax1.set_ylabel(r"$u(x, t) / u^{rms}(x, t)$", color = 'blue')
	ax2 = ax1.twinx()
	ax2.plot(x, du_x_rms[t, :], color = 'red')
	ax2.set_ylabel(r"$\partial_x u(x, t) / \partial_x u^{rms}(x, t)$", color = 'red')

	## FINAL TIME
	t = -1
	ax3.plot(x, u_urms[t, :], color = 'blue')
	ax3.set_xlabel(r"$x$")
	leg3 = mpl.patches.Rectangle((0, 0), 0, 0, alpha = 0.0)
	ax3.legend([leg3], [r"$Iter:({})$".format(t)], handlelength = -0.5, fancybox = True, prop = {'size': 10})
	ax3.set_xlim(0, 2.0 * np.pi)
	ax3.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, 2*np.pi])
	ax3.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
	ax3.set_ylabel(r"$u(x, t) / u^{rms}(x, t)$", color = 'blue')
	ax4 = ax3.twinx()
	ax4.plot(x, du_x_rms[t, :], color = 'red')
	ax4.set_ylabel(r"$\partial_x u(x, t) / \partial_x u^{rms}(x, t)$", color = 'red')

	plt.savefig(output_dir + "/REALSPACE_GRAD_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(N, alpha, beta, k0, iters), format='png', dpi = 400)  
	plt.close()


def plot_spacetime_full(x, u_urms, du_x_rms, time, phases, N, alpha, beta, k0):
    ## Create figure
    fig = plt.figure(figsize = (16, 9), tight_layout=True)
    gs  = GridSpec(4, 2)

    ## Selct time slice
    t_shock = -100
    t_chaos = 500

    ## REAL SPACE - SPACETIME
    ax1 = fig.add_subplot(gs[0:2, 0])
    im1 = ax1.imshow(np.flip(u_urms), cmap = "bwr") 
    ax1.axhline(y = t_shock, xmin = 0, xmax = N, color = 'blue')
    ax1.axhline(y = t_chaos, xmin = 0, xmax = N, color = 'black')
    ax1.set_aspect('auto')
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$t$")
    ax1.set_xticks([0.0, np.ceil(N / 4), np.ceil(2 * N / 4), np.ceil(3 * N / 4), N])
    ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax1.set_yticks([0.0, np.ceil(u_urms.shape[0] / 4), np.ceil(2 * u_urms.shape[0] / 4), np.ceil(3 * u_urms.shape[0] / 4), u_urms.shape[0]])
    ax1.set_yticklabels([r"$1000$", r"$750$", r"$500$", r"$250$", r"$0$"])
    plt.gcf().text(x = 0.001, y = 0.97, s = r"a)", fontsize = 20)
    div1  = make_axes_locatable(ax1)
    cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    cb1   = plt.colorbar(im1, cax = cbax1)
    cb1.set_label(r"$u / u_{rms}$")

    ## PHASES - SPACETIME
    ax2 = fig.add_subplot(gs[0:2, 1])
    im2  = ax2.imshow(np.flipud(np.mod(phases, 2.0*np.pi)), cmap = "Blues", vmin = 0.0, vmax = 2.0 * np .pi)
    ax2.axhline(y = t_shock, xmin = 0, xmax = N, color = 'blue')
    ax2.axhline(y = t_chaos, xmin = 0, xmax = N, color = 'black')
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

    ax3 = fig.add_subplot(gs[2, 0:2])
    ax33 = ax3.twinx()
    ax33.set_ylim(-2.9, 2.9)
    ax33.plot(u_urms[t_shock, :], color = 'blue')
    ax3.set_ylabel(r"$u / u_{rms}$")
    ax3.plot(du_x_rms[t_shock, :], color = 'red')
    ax33.set_ylabel(r"$\partial_x u / \partial_x u_{rms}$", color = 'red')
    ax3.set_ylim(-2.9, 2.9)
    ax3.set_xlim(0, N)
    ax3.set_xticks([0.0, np.ceil(N / 4), np.ceil(2 * N / 4), np.ceil(3 * N / 4), N])
    ax3.set_xticklabels([])
    plt.gcf().text(x = 0.001, y = 0.5, s = r"c)", fontsize = 20)
    
    ax4 = fig.add_subplot(gs[3, 0:2])
    ax44 = ax4.twinx()
    ax44.set_ylim(-2.9, 2.9)
    ax4.set_ylabel(r"$u / u_{rms}$")
    ax44.plot(u_urms[t_chaos, :], color = 'black')
    ax4.set_ylim(-2.9, 2.9)
    ax4.set_xlim(0, N)
    ax4.set_xticks([0.0, np.ceil(N / 4), np.ceil(2 * N / 4), np.ceil(3 * N / 4), N])
    ax4.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    ax4.set_xlabel(r"$x$")
    ax4.plot(du_x_rms[t_chaos, :], color = 'red')
    ax44.set_ylabel(r"$\partial_x u / \partial_x u_{rms}$", color = 'red')
    plt.gcf().text(x = 0.001, y = 0.25, s = r"d)", fontsize = 20)
    
    plt.savefig(output_dir + "/SPACETIME_FULL_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(N, alpha, beta, k0, iters), format='png', dpi = 400)  
    plt.close()




######################
##	     MAIN       ##
######################
if __name__ == '__main__':
	

    ######################
    ##	Get Input Parameters
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

    # filename = "/Runtime_Data_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{}]_TRANS[{}]".format(N, k0, alpha, beta, u0, iters, trans)

    # ######################
    # ##	Input & Output Dir
    # ######################
    # input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Output/Stats"
    # output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/TriadDynamics" + filename
    # if os.path.isdir(output_dir) != True:
    # 	os.mkdir(output_dir)
    # if os.path.isdir(output_dir + '/SNAPS') != True:
    # 	os.mkdir(output_dir + '/SNAPS')

    results_dir = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]".format(N, k0, alpha, beta, u0)
    filename    = "/SolverData_ITERS[{}]_TRANS[{}]".format(iters, trans)

    ######################
    ##	Input & Output Dir
    ######################
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Transfer_Report"




    ######################
    ##	Read in Input File
    ######################
    # HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')

    # # print input file name to screen

    # print("\n\nData File: %s\n" % filename)
    HDFfileData = h5py.File(input_dir + results_dir + filename + '.h5', 'r')

    # print input file name to screen
    # print("\n\nData File: %s.h5\n" % filename)
    print("\n\nData File: {}.h5\n".format(results_dir + filename))


    ######################
    ##	Read in Datasets
    ######################
    phases = HDFfileData['Phases'][:, :]
    amps   = HDFfileData['Amps'][:]
    try:
        time   = HDFfileData['Time'][:]
    except:
        time, step = compute_time(N, alpha, beta, k0, iters, trans)

    ######################
    ##	Preliminary Calcs
    ######################
    ntsteps = phases.shape[0]
    num_osc = phases.shape[1]
    N       = 2 * num_osc - 1 - 1
    kmin    = k0 + 1
    kmax    = amps.shape[0] - 1



    ######################
    ##	Compute Data
    ######################
    ## Compute Real Space vel
    u, u_urms, x, u_z = compute_realspace(amps, phases, N)

    ## Compute phases
    phases_rms = compute_phases_rms(phases)

    ## Compute Real Space vel
    du_x, du_x_rms = compute_gradient(u_z, k0, kmax)



    ######################
    ##	Compute Data
    ######################
    print("Plotting!!\n\n")


    # ## Plot space-time
    # plot_spacetime(x, u_urms, time, phases, N, alpha, beta, k0)


    # ## Plot space-time
    # plot_grad_realspace(x, u_urms, du_x_rms, N, alpha, beta, k0)

    t = 90000
    ## Plot full space-time figure
    plot_spacetime_full(x, u_urms[t:1000 + t, :], du_x_rms[t:1000 + t, :], time[t:1000 + t], phases[t:1000 + t, :], N, alpha, beta, k0)



    print("Finished!!\n\n")
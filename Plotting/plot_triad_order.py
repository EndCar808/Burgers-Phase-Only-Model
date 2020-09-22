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
mpl.rcParams['figure.figsize']    = [10, 8]
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['text.usetex']       = True
mpl.rcParams['font.family']       = 'serif'
mpl.rcParams['font.serif']        = 'Computer Modern Roman'
mpl.rcParams['lines.linewidth']   = 1.25
mpl.rcParams['lines.markersize']  = 6
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
from numba import jit



## Create the triads from the phases
@jit(nopython = True)
def compute_triads(phases, kmin, kmax):
	print("\n...Computing Triads...\n")

	## Variables
	numTriads  = 0;
	k3_range   = int(kmax - kmin + 1)
	k1_range   = int((kmax - kmin + 1) / 2)
	time_steps = phases.shape[0]

	## Create memory space
	triadphase = -10 * np.ones((k3_range, k1_range, time_steps))
	triads     = -10 * np.ones((k3_range, k1_range, time_steps))
	phaseOrder = np.complex(0.0, 0.0) * np.ones((time_steps))
	R          = np.zeros((time_steps))
	Phi        = np.zeros((time_steps))
	
	## Compute the triads
	for k in range(kmin, kmax + 1):
	    for k1 in range(kmin, int(k/2) + 1):
	        triadphase[k - kmin, k1 - kmin, :] = phases[:, k1] + phases[:, k - k1] - phases[:, k]
	        triads[k - kmin, k1 - kmin, :]     = np.mod(triadphase[k - kmin, k1 - kmin, :], 2*np.pi)

	        phaseOrder[:] += np.exp(np.complex(0.0, 1.0)*triads[k - kmin, k1 - kmin, :])
	        numTriads += 1
	
	# Compute Phase-Order params
	R[:]   = np.absolute(phaseOrder[:] / numTriads)
	Phi[:] = np.angle(phaseOrder[:] / numTriads)

	return triads, R, Phi

## Create the Kuramoto-Order from the phases
@jit(nopython = True)
def compute_phase_order(phases, amps, kmin, kmax):
	print("\n...Computing Scale Phase Order...\n")

	## Variables
	k_range     = int(kmax - kmin + 1)
	k1_range    = int((kmax - kmin + 1) / 2)
	time_steps  = phases.shape[0]
	sum_of_amps = 0.0

	phaseOrder = -10 * np.complex(0.0, 0.0) * np.ones((time_steps, k_range))

	## Compute the phase order
	for k in range(kmin, kmax + 1):
		for k_1 in range(kmin + k,  int(2 * kmax)):
			if k_1 < kmax:
				k1 = -kmax + k_1
			else:
				k1 = k_1 - kmax;
			
			if k1 < 0:
				phaseOrder[:, k - kmin] += (amps[np.absolute(k1)] * np.exp(-np.complex(0.0, 1.0) * phases[:, np.absolute(k1)])) * (amps[k - k1] * np.exp(np.complex(0.0, 1.0) * phases[:, k - k1]))
				sum_of_amps += amps[np.absolute(k1)] * amps[k - k1]
			elif k - k1 < 0:
				phaseOrder[:, k - kmin] += (amps[k1] * np.exp(np.complex(0.0, 1.0) * phases[:, k1])) * (amps[np.absolute(k - k1)] * np.exp(-np.complex(0.0, 1.0) * phases[:, np.absolute(k - k1)]))
				sum_of_amps += amps[k1] * amps[np.absolute(k - k1)]
			else:
				phaseOrder[:, k - kmin] += (amps[k1] * np.exp(np.complex(0.0, 1.0) * phases[:, k1])) * (amps[k - k1] * np.exp(np.complex(0.0, 1.0) * phases[:, k - k1]))
				sum_of_amps += amps[k1] * amps[k - k1]

		phaseOrder[:, k - kmin] = (-np.complex(0.0, 1.0) * phaseOrder[:, k - kmin]) / sum_of_amps

	
	return np.absolute(phaseOrder[:, :]), np.angle(phaseOrder[:, :])



def plot_phase_order(R_k, Phi_k, kmin, kmax, t):
	
	r    = 1
	thet = np.linspace(0.0, 2.0*np.pi, 100)

	fig = plt.figure(figsize = (20, 10), tight_layout = False)
	gs  = GridSpec(1, 2)

	## REAL SPACE
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.plot(r * np.cos(thet), r * np.sin(thet), '--', color = 'black')
	ax1.scatter(R_k * np.cos(Phi_k), R_k * np.sin(Phi_k), c = np.arange(len(R_k)), s = 1, cmap = 'viridis')
	ax1.set_xlim(-1, 1)
	ax1.set_ylim(-1, 1)
	ax1.set_xlabel(r"$\Re\left[R_{k} e^{\mathrm{i} \Phi_{k}}\right]$")
	ax1.set_ylabel(r"$\Im\left[R_{k} e^{\mathrm{i} \Phi_{k}}\right]$")
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.spines['bottom'].set_visible(False)
	ax1.spines['left'].set_visible(False)
	ax1.set_xticks([])
	ax1.set_yticks([])

	ax2 = fig.add_subplot(gs[0, 1])
	ax2.plot(r * np.cos(thet), r * np.sin(thet), '--', color = 'black')
	ax2.scatter(R_k * np.cos(Phi_k), R_k * np.sin(Phi_k), c = np.arange(len(R_k)), s = 1, cmap = 'viridis')
	ax2.set_xlim(-1/20, 1/20)
	ax2.set_ylim(-1/20, 1/20)
	ax2.set_xlabel(r"$\Re\left[R_{k} e^{\mathrm{i} \Phi_{k}}\right]$")
	ax2.set_ylabel(r"$\Im\left[R_{k} e^{\mathrm{i} \Phi_{k}}\right]$")
	# ax2.spines['top'].set_visible(False)
	# ax2.spines['right'].set_visible(False)
	# ax2.spines['bottom'].set_visible(False)
	# ax2.spines['left'].set_visible(False)
	# ax2.set_xticks([])
	# ax2.set_yticks([])

	# plt.colorbar()
	# cbar.set_ticks([kmin, kmax/2, kmax])
 #    cbar.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])

	plt.savefig(output_dir + "/ORDER_SNAPS/Order_SNAPS_{:05d}.png".format(t), format='png', dpi = 400)  
	plt.close()





if __name__ == '__main__':
	#########################
	##	Get Input Parameters
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
	filename    = "/LCEData_ITERS[{}]_TRANS[{}]".format(iters, trans)

	######################
	##	Input & Output Dir
	######################
	# input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Output/LCE"
	# output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/TriadDynamics" + filename
	input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
	output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS/" + results_dir

	if os.path.isdir(output_dir) != True:
		os.mkdir(output_dir)
	if os.path.isdir(output_dir + '/SNAPS') != True:
		os.mkdir(output_dir + '/SNAPS')
	if os.path.isdir(output_dir + '/ORDER_SNAPS') != True:
		os.mkdir(output_dir + '/ORDER_SNAPS')



	######################
	##	Read in Input File
	######################
	# HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')
	HDFfileData = h5py.File(input_dir + results_dir + filename + '.h5', 'r')

	# print input file name to screen
	# print("\n\nData File: %s.h5\n" % filename)
	print("\n\nData File: {}.h5\n".format(results_dir + filename))

	######################
	##	Read in Datasets
	######################
	phases = HDFfileData['Phases'][:, :]
	time   = HDFfileData['Time'][:]
	amps   = HDFfileData['Amps'][:]




	######################
	##	Preliminary Calcs
	######################
	ntsteps = len(time);
	num_osc = phases.shape[1];
	N       = 2 * (num_osc - 1);
	kmin    = k0 + 1;
	kmax    = num_osc - 1;



	######################
	##	Triad Data
	######################
	triads, R, Phi = compute_triads(phases, kmin, kmax)
	triads_exist   = 0

	R_k, Phi_k = compute_phase_order(phases, amps, kmin, kmax)

	print(R_k.shape)
	print(Phi_k.shape)


	######################
	##	Plot Data
	######################

	for t in range(len(time)):
		print("Plotting SNAP {}".format(t))
		plot_phase_order(R_k[t, :], Phi_k[t, :], kmin, kmax, t)
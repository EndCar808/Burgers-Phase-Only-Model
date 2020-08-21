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
	# results_dir = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]".format(N, k0, alpha, beta, u0)
	# filename    = "/LCEData_ITERS[{}]_TRANS[{}]".format(iters, trans)
	filename = "/LCE_Runtime_Data_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{}]".format(N, k0, alpha, beta, u0, iters)

	######################
	##	Input & Output Dir
	######################
	input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Output/LCE"
	output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/TriadDynamics" + filename
	# input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
	# output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS/" + results_dir




	######################
	##	Read in Input File
	######################
	HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')
	# HDFfileData = h5py.File(input_dir + results_dir + filename + '.h5', 'r')

	# print input file name to screen
	print("\n\nData File: %s.h5\n" % filename)
	# print("\n\nData File: %s.h5\n".format(results_dir + filename))

	######################
	##	Read in Datasets
	######################
	phases = HDFfileData['Phases'][:, :]
	time   = HDFfileData['Time'][:]



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
	# if 'Triads' in list(HDFfileData.keys()):
	# 	R      = HDFfileData['PhaseOrderR'][:, :]
	# 	Phi    = HDFfileData['PhaseOrderPhi'][:, :]
	# 	triad  = HDFfileData['Triads']
	# 	# Reshape triads
	# 	tdims     = triad.attrs['Triad_Dims']
	# 	triads    = np.array(np.reshape(triad, np.append(triad.shape[0], tdims[0, :])))

	# 	triads_exist = 1
	# else:
	# 	## Call triad function
	# 	triads, R, Phi = compute_triads(phases, kmin, kmax)
	# 	triads_exist = 0

	triads, R, Phi = compute_triads(phases, kmin, kmax)
	triads_exist   = 0

	## CREATE FIGURE
	fig = plt.figure(figsize = (16, 9), tight_layout=True)
	gs  = GridSpec(1, 1)
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.plot(time[:], phases[:, :])
	ax1.set_ylabel(r"$\phi_k (t)$")
	ax1.set_xlim(time[0], time[-1])
	ax1.set_xlabel(r'$t$')

	plt.savefig(output_dir + "/All_Phases.pdf")  
	plt.close()

	fig = plt.figure(figsize = (16, 9), tight_layout=True)
	gs  = GridSpec(1, 1)
	ax1 = fig.add_subplot(gs[0, 0])
	for k in range(num_osc):
		ax1.plot(time[:], phases[:, k])
		ax1.set_ylabel(r"$\phi_{}(t)$".format(k))
		ax1.set_xlim(time[0], time[-1])
		ax1.set_xlabel(r'$t$')
		ax1.legend(r"$k = {}$".format(k))
		plt.savefig(output_dir + "/Phases_k[{}].pdf".format(k))  
		plt.close()


	fig = plt.figure(figsize = (16, 9), tight_layout=True)
	gs  = GridSpec(1, 1)
	ax1 = fig.add_subplot(gs[0, 0])
	for k1 in range(kmin, int(k/2) + 1):
		ax1.plot(time[:], np.transpose(triads[:, k1 - kmin, :]))
		ax1.set_ylabel(r"$\varphi_{k1, k_2}^{k3} (t)$")
		ax1.set_xlim(time[0], time[-1])
		ax1.set_title(r"$k_1 = {}$".format(k1))
		ax1.set_xlabel(r'$t$')
		plt.savefig(output_dir + "/Triads_k1[{}].pdf".format(k))  
		plt.close()
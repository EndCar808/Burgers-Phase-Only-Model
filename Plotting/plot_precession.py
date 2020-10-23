######################
##	Library Imports
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# plt.style.use('seaborn-talk')
mpl.rcParams['figure.figsize']    = [16, 9]
# mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['text.usetex']       = True
mpl.rcParams['font.family']       = 'serif'
mpl.rcParams['font.size']         = 22
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
from scipy.linalg import subspace_angles
np.set_printoptions(threshold=sys.maxsize)
from numba import njit, jit, prange
import itertools

@njit
def compute_indep_triads(phases, N, kmin):

	indep_triads = np.zeros((phases.shape[0], int(N/2 - 3)))

	for i in range(phases.shape[1] - 2*kmin):
		indep_triads[:, i] = phases[:, 2] + phases[:, 2 + i] - phases[:, 4 + i]

	return indep_triads


def compute_precession(triad, num_tsteps, dt, win_size):
    
	df = np.zeros((num_tsteps - 1))

	## Compute derivative
	df = np.diff(triad, 1) / dt

	## Create precession array for this triad
	prec = np.zeros((int(df.shape[0] / win_size)))

	for i, t in enumerate(range(0, num_tsteps - 1, win_size)):
	    prec[i] = np.sum(df[t:t + win_size]) / win_size
	    
	return prec

def plot_trapping(N, a, k, k_lo, k_hi):

	## Read open data file
	results_dir = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]".format(N, k0, a, beta, u0)
	filename    = "/SolverData_ITERS[{}]_TRANS[{}]".format(iters, trans)

	input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
	output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS/" + results_dir


	print("\n\nData File: {}.h5\n".format(results_dir + filename))

	HDFfileData = h5py.File(input_dir + results_dir + filename + '.h5', 'r')


	## Read in datasets
	phases = HDFfileData['Phases'][:, :]
	time   = HDFfileData['Time'][:]
	amps   = HDFfileData['Amps'][:]

	## System Parameters
	num_tsteps = len(time)
	dt = time[1] - time[0]
	num_osc    = amps.shape[0];
	kmin       = k0 + 1;
	kmax       = num_osc - 1;
	dof        = num_osc - kmin

	## Compute independent triads
	indep_triads = compute_indep_triads(phases, N, kmin)

	## Loop over and plot data
	for i in range(k_lo, k_hi):
		plt.figure(figsize = (16, 9))
		if i != 20:
			plt.plot(indep_triads[:, i], indep_triads[:, k], '.')
			ymin, ymax = plt.gca().get_ylim()
			xmin, xmax = plt.gca().get_xlim()
			theta = np.pi / 2
			j = 0
			while(theta * j <= xmax):
			    plt.axvline(x = theta * j, linestyle = '--', color = "black")
			    j += 1
			j = 0
			while(-theta * j >= xmin):
			    plt.axvline(x = -theta * j, linestyle = '--', color = "black")
			    j += 1
			j = 0
			while(theta * j <= ymax):
			    plt.axhline(y = theta * j, linestyle = '--', color = "black")
			    j += 1
			j = 0
			while(-theta * j >= ymin):
			    plt.axhline(y = -theta * j, linestyle = '--', color = "black")
			    j += 1
			plt.xlim(np.amin(indep_triads[:, i]), np.amax(indep_triads[:, i]))
			plt.ylim(np.amin(indep_triads[:, k]), np.amax(indep_triads[:, k]))

			plt.xlabel(r"$\varphi_{{{},{}}}^{{{}}}$".format(2, 2 + i, 4 + i))
			plt.ylabel(r"$\varphi_{{{},{}}}^{{{}}}$".format(2, 2 + k, 4 + k))
				
			plt.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/CLVs" + "/TPhase_Trapping_N[{}]_ALPHA[{:0.3f}]_TRIAD[({},{},{}_{},{},{})].png".format(N, a, 2, 2 + i, 4 + i, 2, 2 + k, 4 + k), format = "png")  
			plt.close()

	return print("Plotted alpha = {:0.3f}".format(a))



def plot_precession(N, k0, a, beta, u0, iters, trans, win_size):

	## Read open data file
	results_dir = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]".format(N, k0, a, beta, u0)
	filename    = "/SolverData_ITERS[{}]_TRANS[{}]".format(iters, trans)

	input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
	output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS/" + results_dir


	print("\n\nData File: {}.h5\n".format(results_dir + filename))

	HDFfileData = h5py.File(input_dir + results_dir + filename + '.h5', 'r')
	


	## Read in datasets
	phases = HDFfileData['Phases'][:, :]
	time   = HDFfileData['Time'][:]
	amps   = HDFfileData['Amps'][:]

	## System Parameters
	num_tsteps = len(time)
	dt = time[1] - time[0]
	num_osc    = amps.shape[0];
	kmin       = k0 + 1;
	kmax       = num_osc - 1;
	dof        = num_osc - kmin

	## Compute independent triads
	indep_triads = compute_indep_triads(phases, N, kmin)

	## Loop over window sizes and compute precession pdf
	for w in win_size:

		print("alpha = {:0.3f}, w = {}".format(a, w))

		## Compute precession for triads
		prec_indep_triads = np.zeros((int((num_tsteps - 1)/ w), int(N/2 - 3)))

		for i in range(num_osc - 2*kmin):
		    prec_indep_triads[:, i] = compute_precession(indep_triads[:, i], num_tsteps, dt, w)


		## Plot Data
		fig = plt.figure(figsize = (16, 9), tight_layout=True)
		gs  = GridSpec(1, 1)


		ax1 = fig.add_subplot(gs[0, 0])
		# hist, bins  = np.histogram(np.ndarray.flatten(prec)[np.nonzero(np.ndarray.flatten(prec))], bins = 100, density = True)
		hist, bins  = np.histogram(np.ndarray.flatten(prec_indep_triads)[np.nonzero(np.ndarray.flatten(prec_indep_triads))], bins = 1000, density = True)
		# hist, bins  = np.histogram(prec_triads, bins = 100, density = True)
		bin_centers = (bins[1:] + bins[:-1]) * 0.5
		ax1.plot(bin_centers, hist)
		ax1.set_xlabel(r"$\Omega_{k_1, k_2}^{k_3}(t, \Delta t)$")
		ax1.legend([r"$\Delta t = {} \delta t$".format(w)], fancybox = True, framealpha = 1, shadow = True)
		ax1.set_ylabel(r"PDF")
		ax1.set_yscale('log')
		ax1.set_title(r"$\alpha = {}$".format(a))

		plt.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/CLVs" + "/TPhase_Precession_N[{}]_ALPHA[{:0.3f}]_WIN[{}].png".format(N, a, w), format = "png")  
		plt.close()

	return print("Plotted alpha = {:0.3f}".format(a))


if __name__ == '__main__':
	#########################
	##  Parameter Space
	#########################
	N        = 256
	k0       = 1
	alpha    = np.array([0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.45])
	beta     = 0.0 
	iters    = 10000000
	trans    = 0
	u0       = "RANDOM"
	win_size = np.array([50, 100, 200, 400, 800])

	#########################
	##  Plot data
	#########################
	## Create Process list  
	# procLim  = 9


	# ## Create iterable group of processes
	# groups_args =  [(mprocs.Process(target = plot_precession, args = (N, k0, a, beta, u0, iters, trans, win_size)) for a in alpha)] * procLim 


	# ## Loop of grouped iterable
	# for procs in zip_longest(*groups_args): 
	# 	processes = []
	# 	for p in filter(None, procs):
	# 	    processes.append(p)
	# 	    p.start()

	# 	for process in processes:
	# 	    process.join()
		    
		    
		    
	plot_trapping(N, alpha[0], 10, 5, 15)
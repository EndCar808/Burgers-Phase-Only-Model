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
mpl.rcParams['figure.figsize'] = [10, 8]
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
mpl.rcParams['lines.linewidth'] = 1.25
mpl.rcParams['lines.markersize'] = 6
from matplotlib import pyplot as plt
import h5py
import sys
import os
import numpy as np
import mpmath as mp
import shutil as shtl
import multiprocessing as mprocs
from threading import Thread
from subprocess import Popen, PIPE
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
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
	phaseOrder = np.complex(0.0, 0.0) * np.ones((time_steps, 1))
	R          = np.zeros((time_steps, 1))
	Phi        = np.zeros((time_steps, 1))
	
	## Compute the triads
	for k in range(kmin, kmax + 1):
	    for k1 in range(kmin, int(k/2) + 1):
	        triadphase[k - kmin, k1 - kmin, :] = phases[:, k1] + phases[:, k - k1] - phases[:, k]
	        triads[k - kmin, k1 - kmin, :]     = np.mod(triadphase[k - kmin, k1 - kmin], 2*np.pi)

	        phaseOrder[:, 0] += np.exp(np.complex(0.0, 1.0)*triads[k - kmin, k1 - kmin, :])
	        numTriads += 1
	
	# Compute Phase-Order params
	R[:, 0]   = np.absolute(phaseOrder[:, 0] / numTriads)
	Phi[:, 0] = np.angle(phaseOrder[:, 0] / numTriads)

	return triads, R, Phi

def fixed_point(NUM_OSC, k0, a, b):
	
	phi  = np.zeros((int(NUM_OSC)))
	
	for i in range(NUM_OSC):
		if i <= k0:
			phi[i] = 0.0;
		elif np.mod(i, 3) == 0:
			phi[i] = np.pi / 2;
		elif np.mod(i, 3) == 1:
			phi[i] = np.pi / 6.0;
		elif np.mod(i, 3) == 2:
			phi[i] = 5.0 * np.pi / 6.0;


	return phi



if __name__ == '__main__':
	#########################
	##	Get Input Parameters
	#########################
	# if (len(sys.argv) != 6):
	#     print("No Input Provided, Error.\nProvide: \nk0\nAlpha\nBeta\nIterations\nN\n")
	#     sys.exit()
	# else: 
	#     k0    = int(sys.argv[1])
	#     alpha = float(sys.argv[2])
	#     beta  = float(sys.argv[3])
	#     iters = int(sys.argv[4])
	#     N     = int(sys.argv[5])
	

	######################
	##	Input & Output Dir
	######################
	
	

	# if os.path.isdir(output_dir) != True:
	# 	os.mkdir(output_dir)



	######################
	##	Read in Input File
	######################
	
	# print input file name to screen
	# print("\n\nData File: %s.h5\n" % filename)


	######################
	##	Read in Datasets
	######################



	

	######################
	##	Preliminary Calcs
	######################




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
	# else:
	# 	## Call triad function
	# 	triads, R, Phi = compute_triads(phases, kmin, kmax)





	############################
	##	Measure Proxy Stability
	############################
	if (len(sys.argv) != 5):
	    print("No Input Provided, Error.\nProvide: \nk0\nAlpha\nBeta\nIterations\nN\n")
	    sys.exit()
	else: 
	    k0    = int(sys.argv[1])
	    alpha = float(sys.argv[2])
	    beta  = float(sys.argv[3])
	    iters = int(sys.argv[4])
	
	input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Output/Solver"
	output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/"

	Nvals = [1024, 2048, 4096]

	


	## CREATE FIGURE
	fig = plt.figure(figsize = (16, 9), tight_layout=True)
	gs  = GridSpec(1, 1)
	## REAL SPACE
	ax1 = fig.add_subplot(gs[0, 0])


	for i, n in enumerate(Nvals):

		filename = "/Runtime_Data_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[NEW]_ITERS[{}]".format(n, k0, alpha, beta, iters)

		HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')


		phases = HDFfileData['Phases'][:, :]
		time   = HDFfileData['Time'][:, :]

		ntsteps = phases.shape[0];
		num_osc = phases.shape[1];
		N       = n;
		kmin    = k0 + 1;
		kmax    = phases.shape[1] - 1
		k0      = kmin - 1;

		## Create fixed point
		fixed_p = fixed_point(num_osc, k0, alpha, beta)


		## Compute distance
		dis = np.zeros((len(time)))
		for t in range(time.shape[0]):
			dis[t] = np.linalg.norm(phases[t, :] - fixed_p[:])

		## Plot
		ax1.plot(time[:, 0], dis)

	ax1.set_yscale('log');
	ax1.set_xlabel(r"$t$")
	ax1.set_ylabel(r"$\|\phi_k(t) - \bar{\phi}_k\|_2$")
	ax1.set_title(r"$\alpha = {}$".format(alpha))
	ax1.legend([r"$N = {}$".format(Nvals[i]) for i in range(len(Nvals))])
	plt.savefig(output_dir + "/Distance_ALPHA[{:0.3f}].pdf".format(alpha))
	plt.close()
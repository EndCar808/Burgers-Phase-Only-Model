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
	phaseOrder = np.complex(0.0, 0.0) * np.ones((time_steps, 1))
	R          = np.zeros((time_steps, 1))
	Phi        = np.zeros((time_steps, 1))
	
	## Compute the triads
	for k in range(kmin, kmax + 1):
	    for k1 in range(kmin, int(k/2) + 1):
	        triadphase[k - kmin, k1 - kmin, :] = phases[:, k1] + phases[:, k - k1] - phases[:, k]
	        triads[k - kmin, k1 - kmin, :]     = np.mod(triadphase[k - kmin, k1 - kmin, :], 2*np.pi)

	        phaseOrder[:, 0] += np.exp(np.complex(0.0, 1.0)*triads[k - kmin, k1 - kmin, :])
	        numTriads += 1
	
	# Compute Phase-Order params
	R[:, 0]   = np.absolute(phaseOrder[:, 0] / numTriads)
	Phi[:, 0] = np.angle(phaseOrder[:, 0] / numTriads)

	return triads, R, Phi



if __name__ == '__main__':
	#########################
	##	Get Input Parameters
	#########################
	if (len(sys.argv) != 6):
	    print("No Input Provided, Error.\nProvide: \nk0\nAlpha\nBeta\nIterations\nN\n")
	    sys.exit()
	else: 
	    k0    = int(sys.argv[1])
	    alpha = float(sys.argv[2])
	    beta  = float(sys.argv[3])
	    iters = int(sys.argv[4])
	    N     = int(sys.argv[5])
	filename = "/LCE_Runtime_Data_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[ALIGNED]_ITERS[{}]".format(N, k0, alpha, beta, iters)

	######################
	##	Input & Output Dir
	######################
	input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Output/LCE"
	output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/TriadDynamics" + filename

	if os.path.isdir(output_dir) != True:
		os.mkdir(output_dir)
	if os.path.isdir(output_dir + '/SNAPS') != True:
		os.mkdir(output_dir + '/SNAPS')


	######################
	##	Read in Input File
	######################
	HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')

	# print input file name to screen
	print("\n\nData File: %s.h5\n" % filename)


	######################
	##	Read in Datasets
	######################
	phases = HDFfileData['Phases'][:, :]
	time   = HDFfileData['Time'][:, :]
	amps   = HDFfileData['Amps'][:, :]



	######################
	##	Preliminary Calcs
	######################
	ntsteps = phases.shape[0];
	num_osc = phases.shape[1];
	N       = 2 * num_osc - 1 - 1;
	kmin    = k0 + 1;
	kmax    = amps.shape[1] - 1
	k0      = kmin - 1;


	triads, R, Phi = compute_triads(phases, kmin, kmax)
	triads_exist = 0


	######################
	##	ColourMaps
	######################
	# Triads Colourmap
	colours = [[1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0.25, 0], [1, 1, 1]]   #located @ 0, pi/2, pi, 3pi/2 and 2pi
	# my_m    = mpl.colors.LinearSegmentedColormap.from_list('my_map', colours, N = kmax)                            # set N to inertial range
	my_m    = mpl.colors.LinearSegmentedColormap.from_list('my_map', cm.hsv(np.arange(255)), N = kmax)                            # set N to inertial range
	my_m.set_under('1.0')
	my_norm = mpl.colors.Normalize(vmin = 0, vmax = 2*np.pi)


	######################
	##	Plot
	######################
	t = 3555
	fig = plt.figure(figsize = (16, 9), tight_layout = True)
	gs  = GridSpec(2, 1)
	ax4 = fig.add_subplot(gs[0:, 0])
	im  = ax4.imshow(triads[:, :, t], cmap = my_m, norm = my_norm)
	kMax = kmax - kmin # Adjusted indices in triads matrix
	kMin = kmin - kmin # Adjusted indices in triads matrix
	ax4.set_xticks([kmin, int((kMax - kMin)/5), int(2 * (kMax - kMin)/5), int(3* (kMax - kMin)/5), int(4 * (kMax - kMin)/5), kMax])
	ax4.set_xticklabels([kmin, int((kmax - kmin)/5), int(2 * (kmax - kmin)/5), int(3* (kmax - kmin)/5), int(4 * (kmax - kmin)/5), kmax])
	ax4.set_yticks([kMin, int((kMax / 2 - kMin)/4), int(2 * (kMax / 2 - kMin)/4), int(3* (kMax / 2 - kMin)/4),  int((kmax)/ 2 - kmin)])
	ax4.set_yticklabels([kmin + kmin, int((kmax / 2 - kmin)/4) + kmin, int(2 * (kmax / 2 - kmin)/4) + kmin, int(3* (kmax / 2 - kmin)/4) + kmin,  int(kmax / 2)])
	ax4.set_xlabel(r'$k_3$', labelpad = 0)
	ax4.set_ylabel(r'$k_1$',  rotation = 0, labelpad = 10)
	ax4.set_xlim(left = kmin - 0.5)
	ax4.set_ylim(bottom = int((kmax)/ 2 - kmin) + 0.5)
	div4  = make_axes_locatable(ax4)
	cax4  = div4.append_axes('right', size = '5%', pad = 0.1)
	cbar4 = plt.colorbar(im, cax = cax4, orientation='vertical')
	cbar4.set_ticks([ 0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
	cbar4.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])


	# Add the annotations
	for k in range(kmin, kmax + 1):
		for k1 in range(kmin, int(k/2) + 1):
			print((k1, k - k1), end = '')
			ax4.text(k, k1, triads[k - kmin, k1 - kmin, t], ha="center", va="center", color="w")
		print()
	plt.savefig(output_dir + "/SNAPS/AnnotatedTriads_t[{}].pdf".format(time[t]))  
	plt.close()
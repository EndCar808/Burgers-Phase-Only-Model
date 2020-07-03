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
import pandas as pd
import os
import time as TIME
import multiprocessing as mprocs
from threading import Thread
from subprocess import Popen, PIPE
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm 
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from numba import jit



#########################
##	Function Inputs
#########################
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


def compute_phases_from_triads(triads, kmin, num_osc):

		new_phi = np.zeros(num_osc)
		tmp_phi = np.zeros(num_osc)

		k_trans = 0
		while kmin + k_trans < num_osc:
			tmp_phi[kmin + k_trans] = 0.0;

			j = 2
			while j * (kmin + k_trans) < num_osc:
				for i in range(1, j):
					k1_indx = (kmin + k_trans) - kmin
					k3_indx = i * (kmin + k_trans) + 1
					tmp_phi[j * (kmin + k_trans)] -= triads[k3_indx, k1_indx]
					# print((k3_indx, k1_indx))
				new_phi[j * (kmin + k_trans)] = tmp_phi[j * (kmin + k_trans)]
				
				j += 1
			
			k_trans += 1

		return new_phi


def print_df(data, row_labs, col_labs):
	return print(pd.DataFrame(data, index = row_labs, columns = col_labs))

if __name__ == '__main__':
	#########################
	##	Get Input Parameters
	#########################
	if (len(sys.argv) != 6):
	    print("No Input Provided, Error.\nProvide:\nk0\nAlpha\nBeta\nIterations\nN\n")
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
	# lce    = HDFfileData['LCE'][:, :]



	######################
	##	Preliminary Calcs
	######################
	ntsteps = phases.shape[0];
	num_osc = phases.shape[1];
	N       = 2 * num_osc - 1 - 1;
	kmin    = k0 + 1;
	kmax    = amps.shape[1] - 1
	k0      = kmin - 1;



	######################
	##	Triad Data
	######################
	if 'Triads' in list(HDFfileData.keys()):
		R      = HDFfileData['PhaseOrderR'][:, :]
		Phi    = HDFfileData['PhaseOrderPhi'][:, :]
		triad  = HDFfileData['Triads'][:, :]
		# Reshape triads
		tdims     = triad.attrs['Triad_Dims']
		triads    = np.reshape(triad, np.append(triad.shape[0], tdims[0, :]))
	else:
		## Call triad function
		triads, R, Phi = compute_triads(phases, kmin, kmax)


	t = 175


	pd.set_option("display.precision", 10)
	print(pd.DataFrame(data=triads[1:,1:, t] - (3.0 * np.pi)/2.0, index = triads[1:, 0, t], columns = triads[0, 1:, t]))


	# fig = plt.figure(figsize = (16, 9), tight_layout=True)
	# gs  = GridSpec(1, 1)

	# ax2 = fig.add_subplot(gs[0, 0])
	# hist, bins  = np.histogram(np.extract(triads[:, :, t] != -10, triads[:, :, t]).flatten(), range = (0 - 0.5 , 2*np.pi + 0.5), bins = 100, density = True);
	# bin_centers = (bins[1:] + bins[:-1]) * 0.5
	# ax2.plot(bins, [1 for i in bins],'.')
	# ax2.plot(bin_centers, hist, '.-')
	# ax2.set_xlim(-0.05, 2 * np.pi+0.05);
	# ax2.set_ylim(1e-4, 15)
	# ax2.axhline(y = 1 / (2 * np.pi), xmin = 0, xmax = 1., ls = '--', c = 'black');
	# ax2.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, 2*np.pi]);
	# ax2.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"]);
	# ax2.set_xlabel(r'$\varphi_{k_1, k_3 - k_1}^{k_3}(t)$');
	# ax2.set_ylabel(r'PDF');
	# ax2.set_yscale('log');
	# ax2.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both');
	# plt.savefig(output_dir + "/HistTriads{}.pdf".format(t))


	# print(phases[t, :])


	# ## Plot
	# fig = plt.figure(figsize = (16, 9), tight_layout=True)
	# gs  = GridSpec(1, 2)

	# ax1 = fig.add_subplot(gs[0, 0])
	# ax1.plot(phases[:, [kmin, 10, 15, 20, 25, kmax]])
	# ax1.legend([kmin, 10, 15, 20, 25, kmax])

	# ax2 = fig.add_subplot(gs[0, 1])
	# ax2.plot(np.mod(phases[:, [kmin, 10, 15, 20, 25, kmax]], 2.0 * np.pi))
	# ax2.set_yticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, 2*np.pi]);
	# ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"]);
	# ax2.set_title(r'$mod_{2\pi}$')
	# ax2.legend([kmin, 10, 15, 20, 25, kmax])

	# plt.savefig(output_dir + "/Phases_v_time.pdf")


	# fig = plt.figure(figsize = (16, 9), tight_layout=True)
	# gs  = GridSpec(1, 1)

	# ax1 = fig.add_subplot(gs[0, 0])
	# hist, bins = np.histogram(phases[:, kmin:].flatten(), bins = 1000)
	# bin_centers = (bins[1:] + bins[:-1]) * 0.5
	# ax1.plot(bin_centers, hist, '.-')
	# ax1.set_xlim(-0.05, 2 * np.pi+0.05);
	# ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, 2*np.pi]);
	# ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"]);

	# plt.savefig(output_dir + "/HistPhases.pdf")


	print()
	print()

	for k3 in range(kmin, kmax + 1):
		for k1 in range(kmin, int(np.floor(k3 / 2) + 1)):
			print("({}, {}, {})\t".format(k1, k3 - k1, k3), end = ' ')
		print()

	print(triads.shape)

	new_phi = compute_phases_from_triads(triads[:, :, t], kmin, num_osc)


	# print()
	# print()

	# print_df(new_phi, np.arange(num_osc), ['0'])
	# print()
	# print_df(np.mod(new_phi, 2.0 * np.pi), np.arange(num_osc), ['0'])
	print(np.mod(-np.pi/6, 2*np.pi))


	## CREATE FIGURE
	fig = plt.figure(figsize = (16, 9), tight_layout=True)
	gs  = GridSpec(1, 1)

	k_trans = 0
	epsilon = 0
	# epsilon = (np.pi / 2 - phases[t, kmin + k_trans]) / (kmin + k_trans)
	ax5  = fig.add_subplot(gs[0, 0])
	ax5.plot(np.arange(kmin, kmax + 1), np.mod(phases[t, kmin:] -  (epsilon * np.arange(kmin, kmax + 1)), 2.0 * np.pi), '.-')
	ax5.set_xlim(kmin - 0.5, kmax + 0.5)
	ax5.set_ylim([0, 2 * np.pi])
	ax5.set_xlabel(r'$k$', labelpad = 0)
	ax5.set_ylabel(r'$\phi_k$', rotation = 0, labelpad = 10)
	ax5.set_yticks([0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi])
	ax5.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
	ax5.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both');

	plt.savefig(output_dir + "/Phases_v_k_[kmin+{}].pdf".format(k_trans))  
	plt.close()




	# ## CREATE FIGURE
	# fig = plt.figure(figsize = (16, 9), tight_layout=True)
	# gs  = GridSpec(1, 1)

	# k_trans = 2
	# epsilon = phases[t, kmin + k_trans] / (kmin + k_trans)
	# ax5  = fig.add_subplot(gs[0, 0])
	# ax5.plot(np.arange(kmin, kmax + 1), np.mod(new_phi[kmin:], 2.0 * np.pi), '.-')
	# ax5.set_xlim(kmin - 0.5, kmax + 0.5)
	# ax5.set_ylim([0, 2 * np.pi])
	# ax5.set_xlabel(r'$k$', labelpad = 0)
	# ax5.set_ylabel(r'$\phi_k$', rotation = 0, labelpad = 10)
	# ax5.set_yticks([0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi])
	# ax5.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
	# ax5.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both');

	# plt.savefig(output_dir + "/Phases_v_k_[NewPhi].pdf")  
	# plt.close()

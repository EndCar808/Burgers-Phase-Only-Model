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
mpl.rcParams['font.size']   = 12
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


######################
##	Create Dataspace
######################
N = [64, 128, 256, 512, 1024]

alpha = np.arange(0.0, 3.5, 0.05)

k0    = [1]
beta  = [0.0]
iters = 400000
trans = 0
trans_alt = 40000

######################
##	Input / Output
######################
input_dir_ali  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Output/LCE"
output_dir     = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/Spectra"
input_dir_zer  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"


for exp in range(5):
	for k in k0:
		for b in beta:
			######################
			##	Get Data
			######################
			max_spec_al       = np.zeros((len(alpha), len(N)))
			max_spec_al_trans = np.zeros((len(alpha), len(N)))
			max_spec_zer      = np.zeros((len(alpha), len(N)))
			max_spec_ran      = np.zeros((len(alpha), len(N)))

			for n in range(0, len(N)):

				spectra_al       = np.zeros((len(alpha), int(N[n] / 2 - k))) 
				spectra_al_trans = np.zeros((len(alpha), int(N[n] / 2 - k)))    
				spectra_zer      = np.zeros((len(alpha), int(N[n] / 2 - k)))    
				spectra_ran      = np.zeros((len(alpha), int(N[n] / 2 - k))) 

				dof = N[n] / 2 - 1 - k 

				for a in range(0, len(alpha)):
					print("(n = {}, k0 = {}, a = {:0.3f}, b = {:0.3f})".format(N[n], k, alpha[a], b))

					# ## ALIGNED
					# filename_al  = "/LCE_Runtime_Data_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[ALIGNED]_ITERS[{}].h5".format(N[n], k, alpha[a], b, iters)
					# file_al      = h5py.File(input_dir_ali + filename_al, 'r')
					# lce_al       = file_al['LCE']
					# spectrum_al  = lce_al[-1, :]
					# spectra_al[a, :]  = spectrum_al
					# max_spec_al[a, n] = lce_al[-1, exp]

					# if k == 1:
					# 	## ALIGNED - TRANS
					# 	filename_al_trans = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{}]_TRANS[{}].h5".format(N[n], k, alpha[a], b, "ALIGNED", iters, trans_alt)
					# 	file_al_trans     = h5py.File(input_dir_zer + filename_al_trans, 'r')
					# 	lce_al_trans      = file_al_trans['LCE']
					# 	spectrum_al_trans = lce_al_trans[718, :]
					# 	spectra_al_trans[a, :]  = spectrum_al_trans
					# 	max_spec_al_trans[a, n] = lce_al_trans[-1, exp]

					# ## ZERO
					# filename_zer = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{}]_TRANS[{}].h5".format(N[n], k, alpha[a], b, "ZERO", iters, trans)
					# file_zer     = h5py.File(input_dir_zer + filename_zer, 'r')
					# lce_zer      = file_zer['LCE']
					# spectrum_zer = lce_zer[-1, :]
					# spectra_zer[a, :]  = spectrum_zer
					# max_spec_zer[a, n] = lce_zer[-1, exp]
					
					## RANDOM
					filename_ran = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{}]_TRANS[{}].h5".format(N[n], k, alpha[a], b, "RANDOM", iters, trans)
					file_ran     = h5py.File(input_dir_zer + filename_ran, 'r')
					lce_ran      = file_ran['LCE']
					spectrum_ran = lce_ran[-1, :]
					spectra_ran[a, :]  = spectrum_ran
					max_spec_ran[a, n] = lce_ran[-1, exp]


			######################
			##	Plot Data
			######################
			## CREATE FIGURE
			# fig = plt.figure(figsize = (16, 9), tight_layout = True)
			# gs  = GridSpec(1, 1)
			# ax = fig.add_subplot(gs[(0, 0)])
			# for n in range(0, len(N)):
			# 	ax.plot(alpha, max_spec_al[:, n], '.-')
			# ax.legend([r"$N = {val}$".format(val = nn) for nn in N])
			# ax.set_title(r"Aligned")
			# ax.set_yscale('symlog')

			# plt.savefig(output_dir + "/N_LargestLyapunov_EXP[{}]_k0[{}]_BETA[{}]_u0[{}].pdf".format(exp + 1, k, b, "Aligned"))
			# plt.close()

			# fig = plt.figure(figsize = (16, 9), tight_layout = True)
			# gs  = GridSpec(1, 1)
			# ax = fig.add_subplot(gs[(0, 0)])
			# for n in range(0, len(N)):
			# 	ax.plot(alpha, max_spec_zer[:, n], '.-')
			# ax.legend([r"$N = {val}$".format(val = nn) for nn in N])
			# ax.set_title(r"Zero")
			# ax.set_yscale('symlog')

			# plt.savefig(output_dir + "/N_LargestLyapunov_EXP[{}]_k0[{}]_BETA[{}]_u0[{}].pdf".format(exp + 1, k, b, "Zero"))
			# plt.close()

			fig = plt.figure(figsize = (10, 8), tight_layout = True)
			gs  = GridSpec(1, 1)
			ax = fig.add_subplot(gs[(0, 0)])
			for n in range(0, len(N)):
				ax.plot(alpha, max_spec_ran[:, n], '.-')
			ax.legend([r"$N = {val}$".format(val = nn) for nn in N])
			ax.set_title(r"Largest Lyapunov Exp. - $k_0 = {}$, $\beta = {}$, $u_0 = {}$".format(k, b, "Random"))
			ax.set_yscale('symlog')

			plt.savefig(output_dir + "/N_LargestLyapunov_EXP[{}]_k0[{}]_BETA[{}]_u0[{}].pdf".format(exp + 1, k, b, "Random"))
			plt.close()


			fig = plt.figure(figsize = (10, 8), tight_layout = True)
			gs  = GridSpec(1, 1)
			ax = fig.add_subplot(gs[(0, 0)])
			for n in range(0, len(N)):
				ax.plot(alpha, max_spec_ran[:, n] / max_spec_ran[0, n], '.-')
			ax.legend([r"$N = {val}$".format(val = nn) for nn in N])
			ax.set_title(r"LLE($\alpha$)/LLE($0$) - $k_0 = {}$, $\beta = {}$, $u_0 = {}$".format(k, b, "Random"))
			ax.set_yscale('log')

			plt.savefig(output_dir + "/N_LargestLyapunov_ZEROEXP_EXP[{}]_k0[{}]_BETA[{}]_u0[{}].pdf".format(exp + 1, k, b, "Random"))
			plt.close()


			# fig = plt.figure(figsize = (16, 9), tight_layout = True)
			# gs  = GridSpec(2, 2)

			# for i, p in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
			# 	if k == 1:
			# 		ax = fig.add_subplot(gs[p])
			# 		ax.plot(alpha, max_spec_al[:, i], '.-')
			# 		ax.plot(alpha, max_spec_al_trans[:, i], '.-')
			# 		ax.plot(alpha, max_spec_zer[:, i], '.-')
			# 		ax.plot(alpha, max_spec_ran[:, i], '.-')
			# 		ax.legend([r"Aligned", r"Aligned - Trans", r"Zero",  r"Random"])
			# 		ax.set_title(r"$N = {}$".format(N[i]))
			# 		ax.set_yscale('symlog')
			# 	else:
			# 		ax = fig.add_subplot(gs[p])
			# 		ax.plot(alpha, max_spec_al[:, i], '.-')
			# 		ax.plot(alpha, max_spec_zer[:, i], '.-')
			# 		ax.plot(alpha, max_spec_ran[:, i], '.-')
			# 		ax.legend([r"Aligned", r"Zero",  r"Random"])
			# 		ax.set_title(r"$N = {}$".format(N[i]))
			# 		ax.set_yscale('symlog')
	
			# plt.suptitle(r"Lyapunov Exponent: No. {}, $k_0 = {}$, $\beta = {}$".format(exp + 1, k, b))
			# plt.savefig(output_dir + "/LargestLyapunov_Exp[{}]_k0[{}]_BETA[{}].pdf".format(exp + 1, k, b))
			# plt.close()


			# ## CREATE FIGURE
			# fig = plt.figure(figsize = (16, 9), tight_layout = True)
			# gs  = GridSpec(2, 2)

			# for i, p in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
			# 	ax = fig.add_subplot(gs[p])
			# 	ax.plot(alpha, np.absolute(max_spec_al[:, i] - max_spec_zer[:, i]), '.-')
			# 	ax.legend([r"Absolute Error"])
			# 	ax.set_title(r"$N = {}$".format(N[i]))
			# 	ax.set_yscale('log')

			# plt.suptitle(r"Error: No. {}, $k_0 = {}$, $\beta = {}$".format(exp + 1, k, b))
			# plt.savefig(output_dir + "/Error_Exp[{}]_k0[{}]_BETA[{}].pdf".format(exp + 1, k, b))
			# plt.close()
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
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
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
import numpy as np
np.set_printoptions(threshold=sys.maxsize)




######################
##	Function Defs
######################
def compute_error(lce):
    error    = np.zeros((lce.shape[0], lce.shape[1]))
    norm_err = np.zeros((lce.shape[0]))
    for i in range(len(lce)):
        if i > 0:
            error[i, :] = np.absolute(lce[i, :] - lce_last) / np.absolute( lce_last)
            norm_err[i] = np.linalg.norm((lce[i, :] - lce_last) /  lce_last)
        lce_last = lce[i,:]

    return error, norm_err




######################
##	     MAIN       ##
######################
if __name__ == '__main__':
	

	######################
	##	Get Input Parameters
	######################
	if (len(sys.argv) != 6):
	    print("No Input Provided, Error.\nProvide k0, Beta and Iteration Values!\n")
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
	output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/Spectra"



	######################
	##	Read in Input File
	######################
	HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')

	# print input file name to screen
	print("\n\nData File: %s\n" % filename)


	######################
	##	Read in Datasets
	######################
	phases = HDFfileData['Phases'][:, :]
	time   = HDFfileData['Time'][:, :]
	lce    = HDFfileData['LCE'][:, :]
	# amps   = HDFfileData['Amps'][:, :]

	######################
	##	Preliminary Calcs
	######################
	ntsteps = phases.shape[0];
	num_osc = phases.shape[1];
	N       = 2 * num_osc - 1 - 1;
	kmin    = k0 + 1;
	kmax    = num_osc - 1
	

	######################
	##	Function Defs
	######################
	error, norm_err = compute_error(lce)
	
	print(lce.shape)
	print(error.shape)
	######################
	##	Plot Data
	######################
	plt.figure()
	plt.plot(norm_err)
	plt.xlabel(r'$t$')
	plt.yscale('log')
	plt.ylabel(r'$\frac{\left\|\Sigma(t_{j}) - \Sigma(t_{j - 1}) \right\|_2}{\left\|\Sigma(t_{j - 1}) \right\|_2}$')
	plt.grid(which = 'both', axis = 'both')
	
	plt.savefig(output_dir + "/LCE_LOGY_ERROR_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[ALIGNED]_ITERS[{}].png".format(N, k0, alpha, beta, iters), format='png', dpi = 400)  
	plt.close()

	plt.figure()
	plt.plot(norm_err)
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$\frac{\left\|\Sigma(t_{j}) - \Sigma(t_{j - 1}) \right\|_2}{\left\|\Sigma(t_{j - 1}) \right\|_2}$')
	plt.grid(which = 'both', axis = 'both')
	
	plt.savefig(output_dir + "/LCE_LOGLOG_ERROR_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[ALIGNED]_ITERS[{}].png".format(N, k0, alpha, beta, iters), format='png', dpi = 400)  
	plt.close()


	plt.figure()
	for i in range(error.shape[1]):
		plt.plot(error[:, i])
	plt.xlabel(r'$t$')
	plt.yscale('log')
	plt.ylabel(r'$\frac{\left\|\lambda_i(t_{j}) - \lambda_i(t_{j - 1})\right\|_2}{\left\|\lambda_i(t_{j - 1})\right\|_2}$')
	plt.grid(which = 'both', axis = 'both')
	# plt.legend(r'$\lamda_{'+str(np.arange(error.shape[1] +1 )+'}$'))
	plt.savefig(output_dir + "/LCE_ERROR_INLCE_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[ALIGNED]_ITERS[{}].png".format(N, k0, alpha, beta, iters), format='png', dpi = 400)  
	plt.close()

	print(error[-1, :])
	plt.errorbar(x = np.arange(1, lce.shape[1] + 1), y = lce[-1, :], yerr = error[-1, :])
	plt.axhline(y = 0, xmin = 0, xmax = 1., ls = '--', c = 'black');
	plt.xlabel(r'$k$');
	plt.ylabel(r'Lyapunov Exponents');
	plt.xlim(0, lce.shape[1] - 1);
	plt.grid(which = 'both', axis = 'both')
	plt.savefig(output_dir + "/LCE_SPECTUM(ERRORBARS)_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[ALIGNED]_ITERS[{}].png".format(N, k0, alpha, beta, iters), format='png', dpi = 400)  
	plt.close()
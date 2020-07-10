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
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm 
import numpy as np 
np.set_printoptions(threshold=sys.maxsize)
from numba import jit



###########################
##	Function Definitions
###########################
# For plotting snapshots
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



@jit(nopython = True)
def compute_velinc(u, rlen):

	# rList  = [int(r) for r in np.arange(1, kmax + 1, kmax/rlen)]
	rList  = [1, kmax]
	du_r   = np.zeros((u.shape[0], u.shape[1], rlen))

	for r_indx, r in enumerate(rList):
		for i in range(u.shape[0]):
			for j in range(u.shape[1]):
					du_r[i, j, r_indx] = u[i, np.mod(j + r, u.shape[1])] - u[i, j]


	return du_r, rList



@jit(nopython = True)
def compute_str_p(du_r, rlen):
	pList = [2, 3, 4, 5, 6]

	str_p     = np.zeros((len(pList), rlen))
	str_p_abs = np.zeros((len(pList), rlen))

	for r in range(rlen):
		for i, p in enumerate(pList):
			p_du_r      = du_r[:, :, r]**p
			str_p[i, r] = np.mean(p_du_r.flatten())
			str_p_abs[i, r] = np.absolute(np.mean(p_du_r.flatten()))

	return str_p, str_p_abs



## Real Space Data
def compute_realspace(amps, phases, N):
	print("\n...Creating Real Space Soln...\n")

	# Create full set of amps and phases
	amps_full   = np.append(amps[0, :], np.flipud(amps[0, 1:-1]))
	phases_full = np.concatenate((phases[:, :], -np.fliplr(phases[:, 1:-1])), axis = 1)

	# Construct modes and realspace soln
	u_z = amps_full * np.exp(np.complex(0.0, 1.0) * phases_full)
	u   = np.real(np.fft.ifft(u_z, axis = 1))

	# Compute normalized realspace soln
	u_rms  = np.sqrt(np.mean(u[:, :]**2, axis = 1))
	u_urms = np.array([u[i, :] / u_rms[i] for i in range(u.shape[0])])

	x = np.arange(0, 2*np.pi, 2*np.pi/N)
	
	return u, u_urms, x, u_z


# # @jit(nopython = True)
# def compute_deriv(u_z, NUM_OSC):

# 	k    = np.arange(NUM_OSC)
# 	dudx = np.zeros((u_z.shape))

# 	k_full = np.append(k, np.flipud(k[1:-1]))

# 	dudx   = np.fft.ifft(np.complex(0.0, 1.0) * k_full * u_z,  axis = 1)

# 	return np.real(dudx)

def compute_gradient(u_z, kmin, kmax):
	print("\nCreating Gradient\n")
	k            = np.concatenate((np.zeros((kmin)), np.arange(kmin, kmax + 1), -np.flip(np.arange(kmin, kmax)), np.zeros((kmin - 1))))
	grad_u_z     = np.complex(0.0, 1.0) * k * u_z
	du_x         = np.real(np.fft.ifft(grad_u_z, axis = 1))
	du_x_rms_tmp = np.sqrt(np.mean(du_x ** 2, axis = 1))
	du_x_rms     = np.array([du_x[i, :] / du_x_rms_tmp[i] for i in range(u_z.shape[0])])

	return du_x, du_x_rms



if __name__ == '__main__':

	#########################
	##	Get Input Parameters
	#########################
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

	filename = "/Runtime_Data_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{}]_TRANS[{}]".format(N, k0, alpha, beta, u0, iters, trans)

	######################
	##	Input & Output Dir
	######################
	input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Output/Stats"
	output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/Stats" + filename

	if os.path.isdir(output_dir) != True:
		os.mkdir(output_dir)
	


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
	# vel_inc = HDFfileData['VelocityIncrements'][:, :]
	vel_inc_s = HDFfileData['VelIncSmall'][:, :]
	vel_inc_m = HDFfileData['VelIncMid'][:, :]
	vel_inc_l = HDFfileData['VelIncLarge'][:, :]
	deriv = HDFfileData['Derivative'][:, :]
	u     = HDFfileData['RealSpace'][:, :]

	######################
	##	Preliminary Calcs
	######################
	ntsteps = time.shape[0];
	num_osc = amps.shape[1];
	N       = 2 * num_osc - 1 - 1;
	kmin    = k0 + 1;
	kmax    = amps.shape[1] - 1
	k0      = kmin - 1;



	######################
	##	Triad Data
	# ######################
	# if 'Triads' in list(HDFfileData.keys()):
	# 	R      = HDFfileData['PhaseOrderR'][:, :]
	# 	Phi    = HDFfileData['PhaseOrderPhi'][:, :]
	# 	triad  = HDFfileData['Triads'][:, :]
	# 	# Reshape triads
	# 	tdims     = triad.attrs['Triad_Dims']
	# 	triads    = np.reshape(triad, np.append(triad.shape[0], tdims[0, :]))
	# else:
	# 	## Call triad function
	# 	triads, R, Phi = compute_triads(phases, kmin, kmax)
		

	######################
	##	Velocity Incrments
	######################
	# VelIncs 
	VelIncsSmall = vel_inc_s.reshape(ntsteps, N)
	VelIncsMid   = vel_inc_m.reshape(ntsteps, N)
	VelIncsLarge = vel_inc_l.reshape(ntsteps, N)
	Deriv        = deriv.reshape(ntsteps, N)

	vflat_s = VelIncsSmall.flatten() / np.std(VelIncsMid.flatten())
	vflat_m = VelIncsMid.flatten() / np.std(VelIncsMid.flatten())
	vflat_l = VelIncsLarge.flatten() / np.std(VelIncsMid.flatten())
	d_flat  = Deriv.flatten() / np.std(Deriv.flatten() )

	Den = False
	numbin = 1000
	hist, bins  = np.histogram(vflat_s, bins = numbin, density = Den);
	bin_centers = (bins[1:] + bins[:-1]) * 0.5
	plt.plot(bin_centers, hist)
	hist, bins  = np.histogram(vflat_m, bins = numbin, density = Den);
	bin_centers = (bins[1:] + bins[:-1]) * 0.5
	plt.plot(bin_centers, hist)
	hist, bins  = np.histogram(vflat_l, bins = numbin, density = Den);
	bin_centers = (bins[1:] + bins[:-1]) * 0.5
	plt.plot(bin_centers, hist)
	hist, bins  = np.histogram(d_flat, bins = numbin, density = Den);
	bin_centers = (bins[1:] + bins[:-1]) * 0.5
	plt.plot(bin_centers, hist)
	plt.xlabel(r"$\delta_ru / \sigma$")
	plt.yscale('symlog')
	plt.legend(["Small", "Mid", "Large", "Deriv"])
	plt.grid(True)

	plt.savefig(output_dir + "/VelInc.png", format='png', dpi = 400)  
	plt.close()



	
	##################################
	rlen = 2
	du_r, rList = compute_velinc( u, rlen)

	plt.figure()
	hist, bins  = np.histogram(d_flat, bins = numbin, density = Den);
	bin_centers = (bins[1:] + bins[:-1]) * 0.5
	plt.plot(bin_centers, hist)
	for i in range(rlen):
		hist, bins  = np.histogram( du_r[:, :, i].flatten() / np.std(du_r[:, :, i].flatten() ), bins = numbin, density = Den);
		bin_centers = (bins[1:] + bins[:-1]) * 0.5
		plt.plot(bin_centers, hist)
	# plt.legend([r"$r = {}$".format(i) for i in range(8)])
	plt.legend([r"$\partial_x u$", r"$r = 2\pi/N$", r"$r = \pi$"])
	plt.yscale('symlog')
	plt.grid(True)

	plt.savefig(output_dir + "/Derivative_and_SmallsclesPyhton.png", format='png', dpi = 400)  
	plt.close()

	# str_p, str_p_abs = compute_str_p(du_r, rlen)

	# plt.figure()
	# for r in range(rlen):
	# 	tmp  = du_r[:, :, r]
	# 	incr = tmp.flatten() / np.std(tmp.flatten())
	# 	print(incr)
	# 	hist, bins  = np.histogram(incr, bins = numbin, density = Den);
	# 	bin_centers = (bins[1:] + bins[:-1]) * 0.5
	# 	plt.plot(bin_centers, hist)
	# plt.xlabel(r"$\delta_ru / \sigma$")
	# plt.yscale('symlog')
	# plt.legend([r"${}$".format(i) for i in range(1, rlen + 1)])

	# plt.savefig(output_dir + "/VelInc_Py.png", format='png', dpi = 400)  
	# plt.close()

	# u_rms  = np.sqrt(np.mean(u[:, :]**2, axis = 1))
	# u_urms = np.array([u[i, :] / u_rms[i] for i in range(u.shape[0])])
	# plt.figure()
	# for p in range(str_p.shape[0]):
	# 	plt.plot(rList, str_p_abs[p, :])
	# plt.legend([r"$p = {}$".format(p) for p in range(str_p.shape[0])])

	# plt.savefig(output_dir + "/AbsStructureFunc.png", format='png', dpi = 400)  
	# plt.close()
	



	##################################
	u_py, u_urms, x, u_z = compute_realspace(amps, phases, N)

	# plt.figure()
	# plt.plot(x, u[100, :] / N)
	# plt.plot(x, u_py[100, :])
	# plt.legend(["C", "Python"])
	# plt.savefig(output_dir + "/RealSpaceCompare.png", format='png', dpi = 400)  
	# plt.close()

	deriv_py deriv_py_rms = compute_deriv(u_z, num_osc)
	deriv_py_flat = deriv_py.flatten() / np.std(deriv_py.flatten())

	print(deriv_py[100, :])

	plt.figure()
	# plt.plot(x, Deriv[100, :])
	plt.plot(x, deriv_py[100, :])
	plt.legend(["C", "Python"])
	plt.savefig(output_dir + "/DerivCompare.png", format='png', dpi = 400)  
	plt.close()
	

	# plt.figure()
	# hist, bins  = np.histogram(d_flat / N, bins = numbin, density = Den);
	# bin_centers = (bins[1:] + bins[:-1]) * 0.5
	# plt.plot(bin_centers, hist)
	# hist, bins  = np.histogram(deriv_py_flat, bins = numbin, density = Den);
	# bin_centers = (bins[1:] + bins[:-1]) * 0.5
	# plt.plot(bin_centers, hist)
	# plt.legend(["C", "Python"])
	# plt.savefig(output_dir + "/DerivPDFCompare.png", format='png', dpi = 400)  
	# plt.close()
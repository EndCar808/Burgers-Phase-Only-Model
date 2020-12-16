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
from numba import njit, prange



# @njit
def compute_zdata(clv, a_k, num_tsteps, dof, numLEs):
    
    z_data       = np.zeros((dof, dof))
    z_data_proj  = np.zeros((dof, dof))
    z_data_wnorm = np.zeros((dof, dof))
    v            = np.zeros((num_tsteps, dof, dof))
    v_wnorm      = np.zeros((num_tsteps, dof, dof))

    K           = np.arange(2.0, float(dof + 2), 1.0)
    K_norm_sqr  = np.linalg.norm(K) ** 2
    K_wnorm_sqr = np.linalg.norm(K * a_k[2:]) ** 2
    
    for t in range(num_tsteps):
        for j in range(numLEs):
            z_data[:, j] += np.square(clv[t, :, j])

            v[t, :, j]       = clv[t, :, j] - (K * (np.dot(clv[t, :, j], K))) / (K_norm_sqr)
            v_wnorm[t, :, j] = (clv[t, :, j] * a_k[2:]) - ((K * a_k[2:]) * (np.dot(clv[t, :, j] * a_k[2:], K * a_k[2:]))) / (K_wnorm_sqr)
            
            v[t, :, j]       = v[t, :, j] / np.linalg.norm(v[t, :, j])
            v_wnorm[t, :, j] = v_wnorm[t, :, j] / np.linalg.norm(v_wnorm[t, :, j])
            
            z_data_proj[:, j]  += np.square(v[t, :, j])
            z_data_wnorm[:, j] += np.square(v_wnorm[t, :, j])
    
    z_data       = z_data / num_tsteps
    z_data_proj  = z_data_proj / num_tsteps
    z_data_wnorm = z_data_wnorm / num_tsteps
    

    return z_data, z_data_proj, z_data_wnorm


# @njit
def compute_entropy(clv, dof):

    H = 0.0

    for i in range(dof):
        H += clv[i] * np.log(clv[i])

    return -H 

# @njit
def compute_centroid(clv, dof, kmin):

    C = 0.0

    for i in range(dof):
        C += clv[i] * (i + kmin)

    return C 



def open_file(N, k0, a, beta, u0, iters, m_end, m_iter, trans, numLEs, func_type):

	
	## Check if file exists and open
	if numLEs == 1:
		## Create filename from data
		filename = input_dir + "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]".format(N, k0, a, beta, u0, iters, m_end, m_iter)

		## Check if file exists and open
		if os.path.exists(filename + "_TRANS[{}]_LEs[{}].h5".format(trans, numLEs)):
		    HDFfileData = h5py.File(filename + "_TRANS[{}]_LEs[{}].h5".format(trans, numLEs), 'r')
		elif os.path.exists(filename + "_TRANS[{}]_LEs[{}].h5".format(trans * 10, numLEs)):
		    HDFfileData = h5py.File(filename + "_TRANS[{}]_LEs[{}].h5".format(trans * 10, numLEs), 'r')
		elif os.path.exists(filename + "_TRANS[{}]_LEs[{}].h5".format(trans * 100, numLEs)):
		    HDFfileData = h5py.File(filename + "_TRANS[{}]_LEs[{}].h5".format(trans * 100, numLEs), 'r')
		elif os.path.exists(filename + "_TRANS[{}]_LEs[{}].h5".format(trans / 100, numLEs)):
		    HDFfileData = h5py.File(filename + "_TRANS[{}]_LEs[{}].h5".format(trans, numLEs), 'r')
		else: 
		    print("File doesn't exist, check parameters!")
		    sys.exit()		
	else:
		## Create filename from data
		filename = input_dir + "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/CLVData_ITERS[{},{},{}]".format(N, k0, a, beta, u0, iters, m_end, m_iter)

		if os.path.exists(filename + "_TRANS[{}].h5".format(trans)):
		    HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans), 'r')
		elif os.path.exists(filename + "_TRANS[{}].h5".format(trans * 10)):
		    HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans * 10), 'r')
		elif os.path.exists(filename + "_TRANS[{}].h5".format(trans / 10)):
		    HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans / 10), 'r')
		elif os.path.exists(filename + "_TRANS[{}].h5".format(trans * 100)):
		    HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans * 100), 'r')
		else: 
		    print("File doesn't exist, check parameters!")
		    sys.exit()


		
	return HDFfileData


def plot_entropy_N(k0, alpha, beta, u0, iters, m_end, m_iter, trans, numLEs, func_type):

	N = np.array([128, 256, 512])

	fig = plt.figure(figsize = (32, 24), tight_layout = False)
	gs  = GridSpec(2, 3)

	ax1 = []
	ax2 = []
	for i in range(3):
		ax1.append(fig.add_subplot(gs[0, i]))
		ax2.append(fig.add_subplot(gs[1, i]))

	entropy       = np.zeros((alpha.shape[0]))
	entropy_proj  = np.zeros((alpha.shape[0]))
	entropy_wnorm = np.zeros((alpha.shape[0]))

	centroid       = np.zeros((alpha.shape[0]))
	centroid_proj  = np.zeros((alpha.shape[0]))
	centroid_wnorm = np.zeros((alpha.shape[0]))
	
	titles = [r"Unadjusted", r"Invariance Removed", r"Invariance Removed and Weighted Norm"]

	for n in N:
		for i, a in enumerate(alpha):

			## ------- Read in Data
			HDFfileData = open_file(n, k0, a, beta, u0, iters, m_end, m_iter, trans, numLEs, func_type)

			## --------- Get System Parameters
			amps    = HDFfileData['Amps'][:]
			kmin    = k0 + 1
			num_osc = amps.shape[0];
			dof     = num_osc - kmin

			## --------- Read in CLV Data
			if numLEs == 1:
				CLVs          = HDFfileData['LargestCLV'][:]
				num_clv_steps = CLVs.shape[0]
				CLV           = np.reshape(CLVs, (num_clv_steps, dof, 1))
			else:
				## Read in data
				CLVs          = HDFfileData['CLVs']
				num_clv_steps = CLVs.shape[0]

				## Reshape the CLV and Angles data
				clv_dims = CLVs.attrs['CLV_Dims']
				CLV      = np.reshape(CLVs, (CLVs.shape[0], dof, dof))

			## --------- Compute time averaged squared vector components
			zdata, zdata_proj, zdata_wnorm = compute_zdata(CLV, amps, num_clv_steps, dof, numLEs)

			## --------- Compute the entropy for the largest CLV
			entropy[i]       = compute_entropy(zdata[:, 0], dof)
			entropy_proj[i]  = compute_entropy(zdata_proj[:, 0], dof)
			entropy_wnorm[i] = compute_entropy(zdata_wnorm[:, 0], dof)

			## --------- Compute the entropy for the largest CLV
			centroid[i]       = compute_centroid(zdata[:, 0], dof, kmin)
			centroid_proj[i]  = compute_centroid(zdata_proj[:, 0], dof, kmin)
			centroid_wnorm[i] = compute_centroid(zdata_wnorm[:, 0], dof, kmin)


		## Plot Proportion of Max entropy
		ax1[0].plot(entropy / np.log(dof))
		ax1[1].plot(entropy_proj / np.log(dof))
		ax1[2].plot(entropy_wnorm / np.log(dof))

		## Plot entropy
		ax2[0].plot(centroid / dof)
		ax2[1].plot(centroid_proj / dof)
		ax2[2].plot(centroid_wnorm / dof)

	for i in range(3):        
		ax1[i].set_xticks(np.arange(0, alpha.shape[0], 10))
		ax1[i].set_xticklabels(np.arange(0.0, 3.5, 0.5))
		ax1[i].set_xlim(0, alpha.shape[0] - 1)
		ax1[i].set_ylim(0, 1)
		ax1[i].set_xlabel(r"$\alpha$")
		ax1[i].set_ylabel(r"Proportion of Max Entropy")
		ax1[i].legend([r"$N = {}$".format(n) for n in N])

		ax2[i].set_xticks(np.arange(0, alpha.shape[0], 10))
		ax2[i].set_xticklabels(np.arange(0.0, 3.5, 0.5))
		ax2[i].set_xlim(0, alpha.shape[0] - 1)
		ax2[i].set_xlabel(r"$\alpha$")
		ax2[i].set_ylabel(r"Vector Centroid")
		ax2[i].legend([r"$N = {}$".format(n) for n in N])
		# ax2[i].set_title(r"Proportion of Max Entropy of Most Unstable Lyapunov Vector")

	plt.tight_layout(rect = (0, 0, 1, 1))
	# fig1.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/CLVs" + "/Largest_CLV_EntropyCentroid_N[VARIED]_ALPHA[VARIED].png", format = "png")  
	fig.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/CLVs" + "/Largest_CLV_EntropyCentroid_N[VARIED]_ALPHA[VARIED].pdf")  


	plt.close()


def plot_largest_clv(N, k0, alpha, beta, u0, iters, m_end, m_iter, trans, numLEs, func_type):

	# Create Plot
	fig1 = plt.figure(figsize = (32, 18), tight_layout = False)
	# fig2 = plt.figure(figsize = (32, 16), tight_layout = False)
	gs1  = GridSpec(3, 3)
	# gs2  = GridSpec(1, 3)

	ax1 = []
	ax2 = []
	ax3 = []
	for i in range(3):
		ax1.append(fig1.add_subplot(gs1[0, i]))
		ax2.append(fig1.add_subplot(gs1[1, i]))
		ax3.append(fig1.add_subplot(gs1[2, i]))

	entropy       = np.zeros((alpha.shape[0]))
	entropy_proj  = np.zeros((alpha.shape[0]))
	entropy_wnorm = np.zeros((alpha.shape[0]))

	centroid       = np.zeros((alpha.shape[0]))
	centroid_proj  = np.zeros((alpha.shape[0]))
	centroid_wnorm = np.zeros((alpha.shape[0]))

	titles = [r"Unadjusted", r"Invariance Removed", r"Invariance Removed and Weighted Norm"]

	for i, a in enumerate(alpha):

		## ------- Read in Data
		HDFfileData = open_file(N, k0, a, beta, u0, iters, m_end, m_iter, trans, numLEs, func_type)

		## --------- Get System Parameters
		amps    = HDFfileData['Amps'][:]
		kmin    = k0 + 1
		num_osc = amps.shape[0];
		dof     = num_osc - kmin

		## --------- Read in CLV Data
		if numLEs == 1:
			CLVs          = HDFfileData['LargestCLV'][:]
			num_clv_steps = CLVs.shape[0]
			CLV           = np.reshape(CLVs, (num_clv_steps, dof, 1))
		else:
			## Read in data
			CLVs          = HDFfileData['CLVs']
			num_clv_steps = CLVs.shape[0]

			## Reshape the CLV and Angles data
			clv_dims = CLVs.attrs['CLV_Dims']
			CLV      = np.reshape(CLVs, (CLVs.shape[0], dof, dof))

		## --------- Compute time averaged squared vector components
		zdata, zdata_proj, zdata_wnorm = compute_zdata(CLV, amps, num_clv_steps, dof, numLEs)

		## --------- Plot Data
		ax1[0].plot(zdata[:, 0] / 10**i)
		ax1[1].plot(zdata_proj[:, 0] / 10**i)
		ax1[2].plot(zdata_wnorm[:, 0] / 10**i)

		## --------- Compute the entropy for the largest CLV
		entropy[i]       = compute_entropy(zdata[:, 0], dof)
		entropy_proj[i]  = compute_entropy(zdata_proj[:, 0], dof)
		entropy_wnorm[i] = compute_entropy(zdata_wnorm[:, 0], dof)

		## --------- Compute the entropy for the largest CLV
		centroid[i]       = compute_centroid(zdata[:, 0], dof, kmin)
		centroid_proj[i]  = compute_centroid(zdata_proj[:, 0], dof, kmin)
		centroid_wnorm[i] = compute_centroid(zdata_wnorm[:, 0], dof, kmin)


	## Plot entropy
	ax2[0].plot(entropy)
	ax2[1].plot(entropy_proj)
	ax2[2].plot(entropy_wnorm)

	## Plot entropy
	ax3[0].plot(centroid)
	ax3[1].plot(centroid_proj)
	ax3[2].plot(centroid_wnorm)

	for i in range(3):        
		## Vectors
		ax1[i].set_xticks(np.arange(0, dof, 6))
		ax1[i].set_xticklabels(np.arange(kmin, num_osc, 6))
		ax1[i].set_xlim(0, dof - 1)
		ax1[i].set_yscale('log')
		ax1[i].set_xscale('log')
		ax1[i].set_yticks([])
		ax1[i].set_yticklabels([])
		ax1[i].set_xlim(kmin, num_osc - 1)
		ax1[i].set_xlabel(r"$k$")
		ax1[i].set_ylabel(r"$\langle |\mathbf{v}|^2\rangle$")
		ax1[i].set_title(titles[i])
		## Entropy
		ax2[i].axhline(y = np.log(dof), linestyle = "--", color = 'black')
		ax2[i].set_xticks(np.arange(0, alpha.shape[0], 10))
		ax2[i].set_xticklabels(np.arange(0.0, 3.5, 0.5))
		ax2[i].set_xlim(0, alpha.shape[0] - 1)
		# ax2[i].set_ylim(0, 1)
		ax2[i].set_xlabel(r"$\alpha$")
		ax2[i].set_ylabel(r"Vector Entropy")
		# ax2[i].set_title(r"Proportion of Max Entropy of Most Unstable Lyapunov Vector")
		## Centroid
		ax3[i].set_xticks(np.arange(0, alpha.shape[0], 10))
		ax3[i].set_xticklabels(np.arange(0.0, 3.5, 0.5))
		ax3[i].set_xlim(0, alpha.shape[0] - 1)
		ax3[i].set_xlabel(r"$\alpha$")
		ax3[i].set_ylabel(r"Vector Centroid")
		# ax3[i].set_title(r"Centroid of Most Unstable Lyapunov Vector")
	fig1.legend([r"$\alpha = {:0.3f}$".format(a) for a in alpha], bbox_to_anchor = (0.0, 1.0), loc = "upper left", fontsize = 'xx-small')


	plt.tight_layout(rect = (0.04, 0, 1, 1))
	# fig1.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/CLVs" + "/Largest_CLV_N[{}]_ALPHA[VARIED].png".format(N), format = "png")  
	fig1.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/CLVs" + "/Largest_CLV_N[{}]_ALPHA[VARIED].pdf".format(N))  


	plt.close('all')




if __name__ == '__main__':
	#########################
	##  Command Line Input
	#########################
	func_type = str(sys.argv[1])


	#########################
	##  Parameter Space
	#########################
	k0     = 1
	alpha  = np.arange(0.00, 3.5, 0.05)
	# alpha  = np.array([0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.45])
	beta   = 0.0
	iters  = 400000
	trans  = 1000
	N      = 512
	u0     = "RANDOM"
	m_end  = 8000
	m_iter = 50
	if len(sys.argv) == 3:
	    numLEs = int(sys.argv[2])
	else:
		numLEs = int(N / 2 - k0 - 1)

	input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"


	# plot_largest_clv(N, k0, alpha, beta, u0, iters, m_end, m_iter, trans, numLEs, func_type)

	plot_entropy_N(k0, alpha, beta, u0, iters, m_end, m_iter, trans, numLEs, func_type)
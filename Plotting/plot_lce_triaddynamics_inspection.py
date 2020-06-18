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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import h5py
import sys
import os
import numpy as np
np.set_printoptions(threshold=sys.maxsize)




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
input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Output"
output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/TriadDynamics/" + filename

if os.path.isdir(output_dir) != True:
	os.mkdir(output_dir)
if os.path.isdir(output_dir + '/SNAPS') != True:
	os.mkdir(output_dir + '/SNAPS')



######################
##	Read in Input File
######################
HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')

# print input file name to screen
print("\n\nData File: %s\n" % filename)


######################
##	Read in Datasets
######################
phases = HDFfileData['Phases']
time   = HDFfileData['Time']
amps   = HDFfileData['Amps']
lce    = HDFfileData['LCE']


######################
##	Preliminary Calcs
######################
ntsteps = phases.shape[0];
num_osc = phases.shape[1];
N       = 2 * num_osc - 1 - 1;
kmin    = np.count_nonzero(amps[0, :] == 0);
kmax    = amps.shape[1] - 1
k0      = kmin - 1;


######################
##	Triad Data
######################
if 'Triads' in list(HDFfileData.keys()):
	R      = HDFfileData['PhaseOrderR']
	Phi    = HDFfileData['PhaseOrderPhi']
	triads = HDFfileData['Triads']
	# Reshape triads
	tdims     = triads.attrs['Triad_Dims']
	triadsnew = np.reshape(triads, np.append(triads.shape[0], tdims[0, :]))
else:
	# Create the triads from the phases
	numTriads  = 0
	kmin       = np.count_nonzero(amps[0, :] == 0)
	kmax       = amps.shape[1] - 1
	triadphase = -10 *np.ones((kmax - kmin + 1, int((kmax - kmin + 1) / 2), 1)) #
	triads     = -10 *np.ones((kmax - kmin + 1, int((kmax - kmin + 1) / 2), 1)) #
	# phaseOrder = np.complex(0.0, 0.0)*np.ones((time.shape[0], 1))
	# R          = np.ones((time.shape[0], 1))
	# Phi        = np.ones((time.shape[0], 1))
	for k in range(kmin, kmax + 1):
	    for k1 in range(kmin, int(k/2) + 1):
	        triadphase[k - kmin, k1 - kmin, 0] = phases[-1, k1] + phases[-1, k - k1] - phases[-1, k]
	        triads[k - kmin, k1 - kmin, 0]     = np.mod(triadphase[k - kmin, k1 - kmin, 0], 2*np.pi)
	        # phaseOrder[:, 0] += np.exp(np.complex(0.0, 1.0)*triads[k - kmin, k1 - kmin, :])
	        # numTriads += 1
	# R[:, 0]   = np.absolute(phaseOrder[:, 0] / numTriads)
	# Phi[:, 0] = np.angle(phaseOrder[:, 0] / numTriads)


# Print Final Time LCEs
print(lce[-1, :])







# ######################
# ##	Plot Triad Data
# ######################
# for k1 in range(k0, 11):
# 	# TRIAD TSERIES
# 	plt.figure()
# 	for i in range(k1, len(triads[:, 0, 0])):
# 		plt.plot(time[:, 0], triads[i, k1, :])
# 	plt.xlabel(r'$t$');
# 	plt.ylabel(r'$\varphi_{{{}, k}}^{{k + {}}}$'.format(str(k1), str(k1)));
# 	plt.grid(b = True);
# 	plt.savefig(output_dir + "/Triad_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_k1[{}].png".format(N, alpha, beta, k0, iters, k1), format='png', dpi = 400)  
# 	plt.close()


# # ######################
# # ##	Plot Summary Data
# # ######################
# if 'Triads' in list(HDFfileData.keys()):
# 	R      = HDFfileData['PhaseOrderR']
# 	Phi    = HDFfileData['PhaseOrderPhi']
# 	triads = HDFfileData['Triads']
# 	# Reshape triads
# 	tdims     = triads.attrs['Triad_Dims']
# 	triadsnew = np.reshape(triads, np.append(triads.shape[0], tdims[0, :]))
# else:
# 	# Create the triads from the phases
# 	numTriads  = 0
# 	kmin       = np.count_nonzero(amps[0, :] == 0)
# 	kmax       = amps.shape[1] - 1
# 	triadphase = -10 *np.ones((kmax - kmin + 1, int((kmax - kmin + 1) / 2), time.shape[0]))
# 	triads     = -10 *np.ones((kmax - kmin + 1, int((kmax - kmin + 1) / 2), time.shape[0]))
# 	phaseOrder = np.complex(0.0, 0.0)*np.ones((time.shape[0], 1))
# 	R          = np.ones((time.shape[0], 1))
# 	Phi        = np.ones((time.shape[0], 1))
# 	for k in range(kmin, kmax + 1):
# 	    for k1 in range(kmin, int(k/2) + 1):
# 	        triadphase[k - kmin, k1 - kmin, :] = phases[:, k1] + phases[:, k - k1] - phases[:, k]
# 	        triads[k - kmin, k1 - kmin, :]     = np.mod(triadphase[k - kmin, k1 - kmin], 2*np.pi)
# 	        phaseOrder[:, 0] += np.exp(np.complex(0.0, 1.0)*triads[k - kmin, k1 - kmin, :])
# 	        numTriads += 1
# 	R[:, 0]   = np.absolute(phaseOrder[:, 0] / numTriads)
# 	Phi[:, 0] = np.angle(phaseOrder[:, 0] / numTriads)

# 	print(triads[:, :, 0])

# # Create Figure with SubAxes
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = [16, 9])
# fig.suptitle(r'$N = {} \quad \alpha = {} \quad \beta = {} \quad k_0 = {}$'.format(N, alpha, beta, k0))
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# # TRIADS PDF
# hist, bins   = np.histogram(np.extract(triads[:, :, :] != -10, triads).flatten(), bins = 500, normed = True);
# bin_centers = (bins[1:]+bins[:-1])*0.5
# ax1.plot(bin_centers, hist, '.-')
# ax1.set_xlim(-0.05, 2 * np.pi+0.05);
# ax1.axhline(y = 1 / (2 * np.pi), xmin = 0, xmax = 1., ls = '--', c = 'black');
# ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, 2*np.pi]);
# ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"]);
# ax1.set_xlabel(r'$\varphi_{k_1, k - k_1}^{k}(t)$');
# ax1.set_ylabel(r'PDF');
# ax1.set_yscale('log');
# ax1.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both');

# # LCE
# ax2.plot(lce[-1, :], '.-');
# ax2.axhline(y = 0, xmin = 0, xmax = 1., ls = '--', c = 'black');
# ax2.set_xlabel(r'$k$');
# ax2.set_ylabel(r'Lyapunov Exponents');
# ax2.set_xlim(0, lce.shape[1] - 1);
# ax2.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both');

# # KURAMOTO PARAMETER
# ax3.plot(time[:, 0], R[:, 0], time[:, 0], Phi[:, 0])
# ax3.set_ylim(0, np.pi);
# ax3.set_yticks([0.0, 0.5,  1.0, np.pi/2.0, 2.0, np.pi]);
# ax3.set_yticklabels([r"$0$", r"$0.5$", r"$1$", r"$\frac{\pi}{2}$", r"$2$", r"$\pi$"]);
# ax3.set_xlabel(r'$t$');
# ax3.legend([r"$\mathcal{R}(t)$", r"$\Phi(t)$"])
# ax3.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both')

# # TRIAD TSERIES
# for i in range(k0, len(triads[:, 0, 0])):
# 	ax4.plot(time[:, 0], triads[i, 0, :])
# ax4.set_xlabel(r'$t$');
# ax4.set_ylabel(r'$\varphi_{{{}, k}}^{{k + {}}}$'.format(str(kmin), str(kmin)));
# ax4.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both');

# plt.savefig(output_dir + "/LCE_DYNAMICS_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(N, alpha, beta, k0, iters), format='png', dpi = 400)  
# plt.close()



# ######################
# ##	Plot Final Time Data
# ######################
# Create Modes
# amps_full   = np.append(amps[0, :], np.flipud(amps[0, :-2]))
# phases_full = np.append(phases[-1, :], -np.flipud(phases[-1, :-2]))
# u_z = amps_full * np.exp(np.complex(0.0, 1.0) * phases_full)
# u   = np.real(np.fft.ifft(u_z))
# x   = np.arange(0, 2*np.pi, 2*np.pi/N)

# plt.plot(x, u / np.sqrt(np.mean(u**2)))
# plt.xlabel(r'$x$')
# plt.ylabel(r'$u(x, T) / u_{rms}(x, T)$')
# plt.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both')
# plt.savefig(output_dir + "/RealSpace_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(N, alpha, beta, k0, iters), format='png', dpi = 400)  
# plt.close()


for i in range(50):
	print("SNAP {}"+format(i))
	plt.plot(np.arange(kmin, kmax + 1), triads[:, i, -1])
	plt.xlabel(r'$k$')
	plt.ylabel(r'$\varphi_{{{}, k}}^{{k + {}}}$'.format(str(kmin + i), str(kmin + i)));
	plt.savefig(output_dir + "/Triad_versus_k_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_k1[{}].png".format(N, alpha, beta, k0, iters, k1 + i), format='png', dpi = 400)  
	plt.close()





######################
##	Plot Real Space Snaps
# ######################
# for i, t in enumerate(time[:, 0]):
# 	print(i)

# 	phases_full = np.append(phases[i, :], -np.flipud(phases[i, :-2]))
# 	u_z = amps_full * np.exp(np.complex(0.0, 1.0) * phases_full)
# 	u   = np.real(np.fft.ifft(u_z))
# 	plt.plot(x, u / np.sqrt(np.mean(u**2)))
# 	plt.xlim(0, 2*np.pi)
# 	plt.ylim(-3.2, 3.2)
# 	plt.xlabel(r'$x$')
# 	plt.ylabel(r'$u(x, T) / u_{rms}(x, T)$')
# 	plt.legend(r'$t = {}$'.format(t))
# 	plt.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both')
# 	plt.savefig(output_dir + "/SNAPS/SNAPS_{:05d}.png".format(i), format='png', dpi = 400)  
# 	plt.close()
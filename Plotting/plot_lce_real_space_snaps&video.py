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
	triadphase = -10 *np.ones((kmax - kmin + 1, int((kmax - kmin + 1) / 2), time.shape[0]))
	triads     = -10 *np.ones((kmax - kmin + 1, int((kmax - kmin + 1) / 2), time.shape[0]))
	phaseOrder = np.complex(0.0, 0.0)*np.ones((time.shape[0], 1))
	R          = np.ones((time.shape[0], 1))
	Phi        = np.ones((time.shape[0], 1))
	for k in range(kmin, kmax + 1):
	    for k1 in range(kmin, int(k/2) + 1):
	        triadphase[k - kmin, k1 - kmin, :] = phases[:, k1] + phases[:, k - k1] - phases[:, k]
	        triads[k - kmin, k1 - kmin, :]     = np.mod(triadphase[k - kmin, k1 - kmin], 2*np.pi)
	        phaseOrder[:, 0] += np.exp(np.complex(0.0, 1.0)*triads[k - kmin, k1 - kmin, :])
	        numTriads += 1
	R[:, 0]   = np.absolute(phaseOrder[:, 0] / numTriads)
	Phi[:, 0] = np.angle(phaseOrder[:, 0] / numTriads)



######################
##	Plot Real Space Snaps
######################
# amps_full   = np.append(amps[0, :], np.flipud(amps[0, :-2]))
# phases_full = np.append(phases[-1, :], -np.flipud(phases[-1, :-2]))
# u_z = amps_full * np.exp(np.complex(0.0, 1.0) * phases_full)
# u   = np.real(np.fft.ifft(u_z))
x   = np.arange(0, 2*np.pi, 2*np.pi/N)



for i, t in enumerate(time[:, 0]):
	print(i)

	phases_full = np.append(phases[i, :], -np.flipud(phases[i, :-2]))
	u_z = amps_full * np.exp(np.complex(0.0, 1.0) * phases_full)
	u   = np.real(np.fft.ifft(u_z))
	plt.plot(x, u / np.sqrt(np.mean(u**2)))
	plt.xlim(0, 2*np.pi)
	plt.ylim(-3.2, 3.2)
	plt.xlabel(r'$x$')
	plt.ylabel(r'$u(x, T) / u_{rms}(x, T)$')
	plt.legend(r'$t = {}$'.format(t))
	plt.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both')
	plt.savefig(output_dir + "/SNAPS/SNAPS_{:05d}.png".format(i), format='png', dpi = 400)  
	plt.close()
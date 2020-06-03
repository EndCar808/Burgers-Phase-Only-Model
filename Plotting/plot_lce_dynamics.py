#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros

######################
##	Library Imports
######################
import matplotlib
# matplotlib.use('TkAgg') # Use this backend for displaying plots in window
matplotlib.use('Agg') # Use this backend for writing plots to file

matplotlib.rcParams['figure.figsize'] = [16, 9]
import h5py
import sys
import os
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
plt.style.use('classic')


######################
##	Read input file
######################
if (len(sys.argv) == 1):
	print("No Input file Directory specified, Error.\n")
	sys.exit()
else:
	HDFfileData = h5py.File(str(sys.argv[1]), 'r')
	

# print input file name to screen
print("\n\nData File: %s\n" % str(sys.argv[1]))


######################
##	Input & Output Dir
######################
input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Output"
output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots"



######################
##	Read In Data
######################
phases = HDFfileData['Phases']
triads = HDFfileData['Triads']
time   = HDFfileData['Time']
amps   = HDFfileData['Amps']
R      = HDFfileData['PhaseOrderR']
Phi    = HDFfileData['PhaseOrderPhi']
lce    = HDFfileData['LCE']


# Reshape triads
tdims     = triads.attrs['Triad_Dims']
triadsnew = np.reshape(triads, np.append(triads.shape[0], tdims[0, :]))


######################
##	Preliminary Calcs
######################
ntsteps = phases.shape[0];
num_osc = phases.shape[1];
N       = 2 * num_osc - 1 - 1;
kmin    = np.count_nonzero(amps[0, :] == 0);
k0      = kmin - 1;


######################
##	Calculate Un mod Triads
######################
triad_2k = np.zeros((len(time), len(phases[0, :]) - 2))

for k in range(2, len(phases[0, :]) - 2):
	triad_2k[:, k] = phases[:, 2] + phases[:, k] - phases[:, k + 2]


######################
##	Preliminary Calcs
######################
# Create Figure with SubAxes
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle(r'$N = {} \quad \alpha = {} \quad \beta = {} \quad k_0 = {}$'.format(N, 1, 0, k0))

# TRIADS PDF
hist, bins   = np.histogram(np.extract(triads[:, :] != -10, triads).flatten(), bins = 500, normed = True);
bin_centers = (bins[1:]+bins[:-1])*0.5
ax1.plot(bin_centers, hist, '.-')
ax1.set_xlim(-0.05, 2 * np.pi+0.05);
ax1.axhline(y = 1 / (2 * np.pi), xmin = 0, xmax = 1., ls = '--', c = 'black');
ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, 2*np.pi]);
ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"]);
ax1.set_xlabel(r'$\varphi_{k_1, k - k_1}^{k}(t)$');
ax1.set_ylabel(r'PDF');
ax1.set_yscale('log');
ax1.grid(b = True);

# LCE
ax2.plot(lce[-1, :], '.-');
ax2.axhline(y = 0, xmin = 0, xmax = 1., ls = '--', c = 'black');
ax2.set_xlabel(r'$k$');
ax2.set_ylabel(r'Lyapunov Exponents');
ax2.set_xlim(0, lce.shape[1] - 1);
ax2.grid(b = True);

# KURAMOTO PARAMETER
ax3.plot(time[:, 0], R[:, 0], time[:, 0], Phi[:, 0])
ax3.set_ylim(0, np.pi);
ax3.set_yticks([0.0, 0.5,  1.0, np.pi/2.0, 2.0, np.pi]);
ax3.set_yticklabels([r"$0$", r"$0.5$", r"$1$", r"$\frac{\pi}{2}$", r"$2$", r"$\pi$"]);
ax3.set_xlabel(r'$t$');
ax3.legend([r"$\mathcal{R}(t)$", r"$\Phi(t)$"])
ax3.grid(True)

# TRIAD TSERIES
for i in range(2, len(triad_2k[0, :])):
	ax4.plot(time[:, 0], triad_2k[:, i])
ax4.set_xlabel(r'$t$');
ax4.set_ylabel(r'$\varphi_{2, k}^{k + 2}$');
ax4.grid(b = True);



# plt.show()

plt.savefig(output_dir + "/LCE_DYNAMICS_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(64, 1.0, 0.0, 1, 500000), format='png', dpi = 800)  
plt.close()
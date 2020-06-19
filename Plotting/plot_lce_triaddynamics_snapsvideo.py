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
lce    = HDFfileData['LCE'][:, :]



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


	## Call triad function
	triads, R, Phi = compute_triads(phases, kmin, kmax)



######################
##	Real Space Data
######################
def compute_realspace(amps, phases, N):
	print("\n...Creating Real Space Soln...\n")

	# Create full set of amps and phases
	amps_full   = np.append(amps[0, :], np.flipud(amps[0, :-2]))
	phases_full = np.concatenate((phases[:, :], -np.fliplr(phases[:, :-2])), axis = 1)

	# Construct modes and realspace soln
	u_z         = amps_full * np.exp(np.complex(0.0, 1.0) * phases_full)
	u           = np.real(np.fft.ifft(u_z, axis = 1))

	# Compute normalized realspace soln
	u_rms       = np.sqrt(np.mean(u[:, :]**2, axis = 1))
	u_urms      = np.array([u[i, :] / u_rms[i] for i in range(u.shape[0])])

	x   = np.arange(0, 2*np.pi, 2*np.pi/N)
	
	return u, u_urms, x


## Call realspace function
u, u_urms, x = compute_realspace(amps, phases, N)




######################
##	ColourMaps
######################
# Triads Colourmap
colours = [[1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0.25, 0], [1, 1, 1]]   #located @ 0, pi/2, pi, 3pi/2 and 2pi
my_m    = mpl.colors.LinearSegmentedColormap.from_list('my_map', colours, N = kmax)                            # set N to inertial range
my_norm = mpl.colors.Normalize(vmin = 0, vmax = 2*np.pi)

# Phases Colourmap
myhsv   = cm.hsv(np.arange(255))
norm    = mpl.colors.Normalize(vmin = 0.0, vmax = 2.0*np.pi)
my_mhsv = mpl.colors.LinearSegmentedColormap.from_list('my_map', myhsv, N = kmax) # set N to inertial range
m       = cm.ScalarMappable( norm = norm, cmap = my_mhsv)                         # map the values to rgba tuple




######################
##	Plot Data
######################
def plot_snaps(i, x, u_urms, time, triads, kmin, kmax, phases):
	print("SNAP {}\n".format(i))

	## CREATE FIGURE
	fig = plt.figure(figsize = (16, 9), tight_layout=True)
	gs  = GridSpec(4, 2)


	## REAL SPACE
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.plot(x, u_urms)
	ax1.set_xlim(0, 2*np.pi)
	ax1.set_ylim(-3.2, 3.2)
	ax1.set_xlabel(r'$x$')
	ax1.set_ylabel(r'$u(x, t) / u_{rms}(x, t)$')
	ax1.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, 2*np.pi]);
	ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"]);
	leg1 = mpl.patches.Rectangle((0, 0), 0, 0, alpha = 0.0)
	ax1.legend([leg1], [r"$T=({:04.3f})$".format(time)], handlelength = -0.5, fancybox = True, prop = {'size': 10})
	ax1.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both')

	## PDF
	ax2 = fig.add_subplot(gs[0, 1])
	hist, bins  = np.histogram(np.extract(triads != -10, triads).flatten(), bins = int(kmax / 2), density = True);
	bin_centers = (bins[1:] + bins[:-1]) * 0.5
	ax2.plot(bin_centers, hist, '.-')
	ax2.set_xlim(-0.05, 2 * np.pi+0.05);
	ax2.set_ylim(1e-4, 15)
	ax2.axhline(y = 1 / (2 * np.pi), xmin = 0, xmax = 1., ls = '--', c = 'black');
	ax2.set_xticks([0.0, np.pi/2.0, np.pi, 1.5*np.pi, 2*np.pi]);
	ax2.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"]);
	ax2.set_xlabel(r'$\varphi_{k_1, k_3 - k_1}^{k_3}(t)$');
	ax2.set_ylabel(r'PDF');
	ax2.set_yscale('log');
	ax2.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both');

	## PHASES - BARPLOT
	ax3  = fig.add_subplot(gs[1, :])
	div3 = make_axes_locatable(ax3)
	cax3 = div3.append_axes('right', size = '5%', pad = 0.1)
	ax3.bar(x = np.arange(kmin, kmax + 1), height = 1, width = 1, align = 'center', color = m.to_rgba(phases), linewidth =  0)
	ax3.set_xlim(kmin, kmax)
	ax3.set_yticks([ ])
	ax3.set_yticklabels([ ])
	ax3.set_ylim([0, 1])
	ax3.set_xlabel(r'$k$', labelpad = 0)
	ax3.set_ylabel(r'$\phi_k$', rotation = 0, labelpad = 10)
	cbar3 = mpl.colorbar.ColorbarBase(cax3, cmap = my_mhsv, norm = norm)
	cbar3.set_ticks([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
	cbar3.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
    
	## TRIADS
	ax4 = fig.add_subplot(gs[2:, :])
	im  = ax4.imshow(np.transpose(triads), cmap = my_m, norm = my_norm)
	ax4.set_xlim(left = kmin)
	ax4.set_xlabel(r'$k_3$', labelpad = 0)
	ax4.set_ylabel(r'$k_1$',  rotation = 0, labelpad = 10)
	div4  = make_axes_locatable(ax4)
	cax4  = div4.append_axes('right', size = '5%', pad = 0.1)
	cbar4 = plt.colorbar(im, cax = cax4, orientation='vertical')
	cbar4.set_ticks([ 0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
	cbar4.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])


	plt.savefig(output_dir + "/SNAPS/Triad_SNAPS_{:05d}.png".format(i), format='png', dpi = 400)  
	plt.close()



for i in range(time.shape[0]):
	plot_snaps(i, x, u_urms[i, :], time[i, 0], triads[:, :, i], kmin, kmax, phases[i, kmin:])




######################
##	Make Video From Snaps
######################
framesPerSec = 25
inputFile    = output_dir + "/SNAPS/Triad_SNAPS_%05d.png"
videoName    = output_dir + "/SNAPS/Triad_Dynamics.mp4"
cmd = "ffmpeg -r {} -f image2 -s 1920x1080 -i {} -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)


process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
[runCodeOutput, runCodeErr] = proc.communicate()
print(runCodeOutput)
print(runCodeErr)
proc.wait()


print("Finished making video...")
print("Video Location...")
print("\n" + videoName + "\n")



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
from subprocess import Popen, PIPE
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
print("\n\nData File: %s\n" % filename)


######################
##	Read in Datasets
######################
phases = HDFfileData['Phases']
time   = HDFfileData['Time']
amps   = HDFfileData['Amps']




######################
##	Plot Real Space Snaps
######################
print("\nCreating Real Space Soln\n")
amps_full   = np.append(amps[0, :], np.flipud(amps[0, :-2]))
phases_full = np.concatenate((phases[:, :], -np.fliplr(phases[:, :-2])), axis = 1)
u_z         = amps_full * np.exp(np.complex(0.0, 1.0) * phases_full)
u           = np.real(np.fft.ifft(u_z, axis = 1))
u_rms       = np.sqrt(np.mean(u[:, :]**2, axis = 1))
u_urms      = np.array([u[i, :] / u_rms[i] for i in range(u.shape[0])])
x   = np.arange(0, 2*np.pi, 2*np.pi/N)
print("\nFinished...Starting Plotting\n")


for i, t in enumerate(time[:, 0]):
	print("SNAP {}\n".format(i))
	
	plt.plot(x, u_urms[i, :])
	plt.xlim(x[0], x[-1])
	plt.ylim(-3.2, 3.2)
	plt.xlabel(r'$x$')
	plt.ylabel(r'$u(x, t) / u_{rms}(x, t)$')
	leg1 = mpl.patches.Rectangle((0, 0), 0, 0, alpha = 0.0)
	plt.legend([leg1], [r"$T=({:04.3f})$".format(time[0, i])], handlelength = -0.5, fancybox = True, prop = {'size': 10})
	plt.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both')
	plt.savefig(output_dir + "/SNAPS/RealSpace_SNAPS_{:05d}.png".format(i), format='png', dpi = 400)  
	plt.close()



#####################
#	Make Video From Snaps
#####################
framesPerSec = 25
inputFile    = output_dir + "/SNAPS/RealSpace_SNAPS_%05d.png"
videoName    = output_dir + "/SNAPS/Real_Space.mp4"
cmd = "ffmpeg -r {} -f image2 -s 1920x1080 -i {} -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(framesPerSec, inputFile, videoName)


print(inputFile)
print("\n\n")
print(videoName)
print("\n\n")
print(cmd)

process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
[runCodeOutput, runCodeErr] = proc.communicate()
print(runCodeOutput)
print(runCodeErr)
proc.wait()
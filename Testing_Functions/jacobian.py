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
from matplotlib import pyplot as plt
import h5py
import sys
import os
import numpy as np
import mpmath as mp
import shutil as shtl
import multiprocessing as mprocs
from threading import Thread
from subprocess import Popen, PIPE
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec



######################
##	Function Defs
######################
def fill_data(NUM_OSC, k0, a, b, ic):
	
	amp  = mp.zeros(int(NUM_OSC), int(1))
	phi  = mp.zeros(int(NUM_OSC), int(1))
	u_z  = mp.zeros(int(NUM_OSC), int(1))

	if ic == 'JFM':
		for i in range(NUM_OSC):
			if i <= k0:
				amp[i] = mp.mpf(0.0);
				phi[i] = mp.mpf(0.0);
				u_z[i] = mp.mpc(real = '0.0', imag = '0.0')	
			else:
				amp[i] = mp.power(mp.mpf(i), mp.mpf(-a)) * mp.exp(mp.mpf(-b) * mp.power((mp.mpf(i) / mp.mpf((NUM_OSC - 1)/2)) ,2));
				phi[i] = mp.pi / mp.mpf(2);

			u_z[i] = amp[i] * mp.expj(phi[i])
	elif ic == 'NEW':
		for i in range(NUM_OSC):
			if i <= k0:
				amp[i] = mp.mpf(0.0);
				phi[i] = mp.mpf(0.0);
				u_z[i] = mp.mpc(real = '0.0', imag = '0.0')
			elif np.mod(i, 3) == 0:
				amp[i] = mp.power(mp.mpf(i), mp.mpf(-a)) * mp.exp(mp.mpf(-b) * mp.power((mp.mpf(i) / mp.mpf((NUM_OSC - 1)/2)) ,2));
				phi[i] = mp.pi / mp.mpf(2);
			elif np.mod(i, 3) == 1:
				amp[i] = mp.power(mp.mpf(i), mp.mpf(-a)) * mp.exp(mp.mpf(-b) * mp.power((mp.mpf(i) / mp.mpf((NUM_OSC - 1)/2)) ,2));
				phi[i] = mp.pi / mp.mpf(6);
			elif np.mod(i, 3) == 2:
				amp[i] = mp.power(mp.mpf(i), mp.mpf(-a)) * mp.exp(mp.mpf(-b) * mp.power((mp.mpf(i) / mp.mpf((NUM_OSC - 1)/2)) ,2));
				phi[i] = (mp.mpf(5) * mp.pi) / mp.mpf(6);

			u_z[i] = amp[i] * mp.expj(phi[i])

	return u_z, amp, phi	


def convolution(u_z, NUM_OSC, kmin):
	
	conv = mp.zeros(int(NUM_OSC), int(1))

	for k in range(kmin, NUM_OSC):
		for k_1 in range(1 + k, 2 * NUM_OSC):
			if k_1 < NUM_OSC:
				k1 = -NUM_OSC + k_1
			else:
				k1 = k_1 - NUM_OSC 

			if k1 < 0:
				conv[k] += mp.conj(u_z[abs(k1)]) * u_z[k - k1]
			elif k - k1 < 0:
				conv[k] += mp.conj(u_z[abs(k - k1)]) * u_z[k1]
			else:
				conv[k] += u_z[abs(k - k1)] * u_z[k1]
		

	return conv


def compute_jacobian(u_z, NUM_OSC, kmin, kmax, k0):

	jac = mp.zeros(int(NUM_OSC - kmin))
	
	conv = convolution(u_z, NUM_OSC, k0)

	for k in range(kmin, NUM_OSC):

		for kp in range(kmin, NUM_OSC):

			if k == kp:
				if k + kp <= kmax:
					jac[k - kmin, kp - kmin] = mp.mpf(k) * mp.im((u_z[kp] * mp.conj(u_z[k + kp])) / mp.conj(u_z[k]))
					jac[k - kmin, kp - kmin] -= mp.mpf(k / 2) * mp.im(conv[kp] / u_z[k])
				else:
					jac[k - kmin, kp - kmin] = mp.mpf(0)
					jac[k - kmin, kp - kmin] -= mp.mpf(k / 2) * mp.im(conv[kp] / u_z[k])
			else:
				if k + kp > kmax:
					if k - kp < -k0:
						jac[k - kmin, kp - kmin] = mp.mpf(k) * mp.im((u_z[kp] * mp.conj(u_z[abs(k - kp)])) / u_z[k])
					elif k - k0 > k0:
						jac[k - kmin, kp - kmin] = mp.mpf(k) * mp.im((u_z[kp] * u_z[k - kp]) / u_z[k])
					else:
						jac[k - kmin, kp - kmin] = mp.mpf(0)
				else:
					if k - kp < -k0:
						jac[k - kmin, kp - kmin] = mp.mpf(k) * mp.im((u_z[kp] * mp.conj(u_z[abs(k - kp)])) / u_z[k]) + mp.mpf(k) * mp.im((u_z[kp] * mp.conj(u_z[k + kp])) / mp.conj(u_z[k]))
					elif k - k0 > k0:
						jac[k - kmin, kp - kmin] = mp.mpf(k) * mp.im((u_z[kp] * u_z[k - kp]) / u_z[k]) + mp.mpf(k) * mp.im((u_z[kp] * mp.conj(u_z[k + kp])) / mp.conj(u_z[k]))
					else:
						jac[k - kmin, kp - kmin] = mp.mpf(k) * mp.im((u_z[kp] * mp.conj(u_z[k + kp])) / mp.conj(u_z[k]))


	return jac

def compute_stability_verus_alpha(nvals, Alphavals):
	Nvals = nvals
	print(Nvals)

	alpha = Alphavals
	print(alpha)

	prop = np.zeros((len(Nvals), len(alpha)))

	for l, n in enumerate(Nvals):

		f = h5py.File("EigenValuesData_N[{}].hdf5".format(Nvals[l]), "w")
		for j, aa in enumerate(alpha):
			######################
			##	Setup Vars
			######################
			N = int(n)
			NUM_OSC = int(N / 2 + 1)

			## Set precision
			mp.dps = int(N / 2)

			k0 = 2
			a  = mp.mpf(aa)
			b  = mp.mpf(0.0)
			kmin = k0 + 1
			kmax = NUM_OSC - 1

			print(kmin)

			
			print("\n|----------------------------------------------------------------------------------------------------------------------------|")
			print("|----------------------------------------------N = {}  Alpha = {:0.3f}---------------------------------------------------------|\n".format(Nvals[l], alpha[j]))
			######################
			##	Get Data
			######################
			u_z, amp, phi = fill_data(NUM_OSC, k0, a, b, 'NEW')

			# print(u_z)
			# print()
			# print(amp)
			# print()
			# print(phi)

			######################
			##	Compute Jacobian
			######################
			jac = compute_jacobian(u_z, NUM_OSC, kmin, kmax, k0)

			E, ER = mp.eig(jac)
			E, ER = mp.eig_sort(E, ER, f = lambda x: -mp.re(x))  # sort in descending order
			
			## Print order eigs to screen
			mp.nprint(E)
			print()

			## Extract real and imaginary parts
			real_eigs = np.zeros(len(E))
			imag_eigs = np.zeros(len(E))
			for i in range(len(E)):
				real_eigs[i] = E[i].real
				imag_eigs[i] = E[i].imag


			# Write the eigs for later analysis
			f.create_dataset("Real_ALPHA[{:0.3f}]".format(aa), data = real_eigs)
			f.create_dataset("Imag_ALPHA[{:0.3f}]".format(aa), data = imag_eigs)

			# Remove zero eigenvalue due to translation invariance
			min_indx = np.argmin(np.absolute(real_eigs))
			non_zero_eigs = np.extract(real_eigs != real_eigs[min_indx], real_eigs)

			# Print Non-zero eigs to screen
			print(non_zero_eigs)
			print()

			## Compute Proportion of unstable eigenvalues
			pos_eigs = np.extract(non_zero_eigs > 0, non_zero_eigs)
			prop[l, j] = len(pos_eigs) / len(non_zero_eigs)


			print("|----------------------------------------------------------------------------------------------------------------------------|")
			print("|----------------------------------------------------------------------------------------------------------------------------|\n")

	
	######################
	##	Plot Result Data
	######################
	fig = plt.figure(figsize = (16, 9), tight_layout=True)
	gs  = GridSpec(1, 1)

	ax1 = fig.add_subplot(gs[0, 0])
	for n in range(len(Nvals)):
		ax1.plot(alpha, prop[n, :], '.-')
	ax1.set_ylim(0.0 - 0.01, 1.0)
	ax1.set_xlim(alpha[0], alpha[-1])
	ax1.set_xlabel(r"$\alpha$")
	ax1.set_ylabel(r"Proportion of Unstable Eigenvalues")
	ax1.legend([r"$N = {}$".format(Nvals[i]) for i in range(len(Nvals))])
	ax1.grid(True)

	plt.savefig(os.getcwd() + "/PropUnstable_Eigenvals_v_alpha.pdf")
	plt.close()


def compute_stability_verus_N(nvals, Alphaval):
	Nvals = nvals
	print(Nvals)

	alpha = Alphaval
	print(alpha)

	prop = np.zeros((len(Nvals)))

	# Create file
	f = h5py.File("EigenValuesData_v_N_ALPHA[{:0.3f}].hdf5".format(alpha), "w")

	for l, n in enumerate(Nvals):		
			######################
			##	Setup Vars
			######################
			N = int(n)
			NUM_OSC = int(N / 2 + 1)

			## Set precision
			mp.dps = int(N / 2)

			k0 = 2
			a  = mp.mpf(alpha)
			b  = mp.mpf(0.0)
			kmin = k0 + 1
			kmax = NUM_OSC - 1

			print(kmin)

			
			print("\n|----------------------------------------------------------------------------------------------------------------------------|")
			print("|----------------------------------------------N = {}  Alpha = {:0.3f}---------------------------------------------------------|\n".format(Nvals[l], alpha))
			######################
			##	Get Data
			######################
			u_z, amp, phi = fill_data(NUM_OSC, k0, a, b, 'NEW')

			# print(u_z)
			# print()
			# print(amp)
			# print()
			# print(phi)

			######################
			##	Compute Jacobian
			######################
			jac = compute_jacobian(u_z, NUM_OSC, kmin, kmax, k0)

			E, ER = mp.eig(jac)
			E, ER = mp.eig_sort(E, ER, f = lambda x: -mp.re(x))  # sort in descending order
			
			## Print order eigs to screen
			mp.nprint(E)
			print()

			## Extract real and imaginary parts
			real_eigs = np.zeros(len(E))
			imag_eigs = np.zeros(len(E))
			for i in range(len(E)):
				real_eigs[i] = E[i].real
				imag_eigs[i] = E[i].imag


			# Write the eigs for later analysis
			f.create_dataset("Real_N[{:0.3f}]".format(Nvals[l]), data = real_eigs)
			f.create_dataset("Imag_N[{:0.3f}]".format(Nvals[l]), data = imag_eigs)

			# Remove zero eigenvalue due to translation invariance
			min_indx = np.argmin(np.absolute(real_eigs))
			non_zero_eigs = np.extract(real_eigs != real_eigs[min_indx], real_eigs)

			# Print Non-zero eigs to screen
			print(non_zero_eigs)
			print()

			## Compute Proportion of unstable eigenvalues
			pos_eigs = np.extract(non_zero_eigs > 0, non_zero_eigs)
			prop[l] = len(pos_eigs) / len(non_zero_eigs)


			print("|----------------------------------------------------------------------------------------------------------------------------|")
			print("|----------------------------------------------------------------------------------------------------------------------------|\n")

	
	######################
	##	Plot Result Data
	######################
	fig = plt.figure(figsize = (16, 9), tight_layout=True)
	gs  = GridSpec(1, 1)

	ax1 = fig.add_subplot(gs[0, 0])
	for n in range(len(Nvals)):
		ax1.plot(Nvals, prop[n], '.-')
	ax1.set_ylim(0.0 - 0.01, 1.0)
	ax1.set_xlim(Nvals[0], Nvals[-1])
	ax1.set_xlabel(r"$\N$")
	ax1.set_ylabel(r"Proportion of Unstable Eigenvalues")
	# ax1.legend([r"$N = {}$".format(Nvals[i]) for i in range(len(Nvals))])
	ax1.title(r"Proportion of Unstable Eigenvalues - $\alpha = {}$".format(alpha))
	ax1.grid(True)

	plt.savefig(os.getcwd() + "/PropUnstable_Eigenvals_v_N.pdf")
	plt.close()


######################
##	Main
######################
if __name__ == '__main__':

	Nvals = [32, 64, 128, 256, 512, 1024, 2048, 8096]

	alpha = 1.5

	compute_stability_verus_N(Nvals, alpha)
	

	
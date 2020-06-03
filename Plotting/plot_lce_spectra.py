#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros

######################
##	Library Imports
######################
import matplotlib as mpl
mpl.use('TkAgg') # Use this backend for displaying plots in window
# matplotlib.use('Agg') # Use this backend for writing plots to file

mpl.rcParams['figure.figsize'] = [16, 9]
import h5py
import sys
import os
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
plt.style.use('classic')


######################
##	Create dataspace arrays
######################
# N = 2**np.arange(4, 9)
N = [64]
alpha = np.arange(0.0, 2.55, 0.05)


######################
##  Get Input Values
######################
if (len(sys.argv) != 4):
    print("No Input Provided, Error.\nProvide k0, Beta and Iteration Values!\n")
    sys.exit()
else: 
    k0    = int(sys.argv[1])
    beta  = float(sys.argv[2])
    iters = int(sys.argv[3])

######################
##  Input & Output Dirs
######################
input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Output"
output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots"

######################
##	Allocate Memory
######################
num_pos_lce     = np.zeros((len(N), len(alpha)))
prop_pos_lce    = np.zeros((len(N), len(alpha)))
spectrum_sum    = np.zeros((len(N), len(alpha)))
kaplan_york_dim = np.zeros((len(N), len(alpha)))


######################
##	Create Colourmap
######################
norm = mpl.colors.Normalize(vmin = np.array(alpha).min(), vmax = np.array(alpha).max())
cmap = mpl.cm.ScalarMappable(norm = norm, cmap = mpl.cm.viridis)
cmap.set_array([])



######################
##	Loop Over Dataspace
######################
for n in range(0, len(N)):

    spectra = np.zeros((len(alpha), int(N[n] / 2 - k0)))

    for a in range(0, len(alpha)):

        # Read in data
        filename = "/LCE_Runtime_Data_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[ALIGNED]_ITERS[{}].h5".format(N[n], k0, alpha[a], beta, iters)
        file     = h5py.File(input_dir + filename, 'r')
        
        # Extract LCE Data
        lce = file['LCE']
        
        # Extract final state
        spectrum = lce[-1, :]
        spectra[a, :] = spectrum
        
        # find the zero mode
        minval  = np.amin(np.absolute(spectrum))
        minindx = np.where(np.absolute(spectrum) == minval)
        
        # Extract the zero mode
        non_zero_spectrum = np.delete(spectrum, minindx, 0)
        
        # Get the number of positive and negative exponents
        pos_lce = np.extract(non_zero_spectrum > 0, non_zero_spectrum)
        neg_lce = np.extract(non_zero_spectrum < 0, non_zero_spectrum)
        
        # Get the proportion of positive
        num_pos_lce[n, a]  = len(pos_lce)
        prop_pos_lce[n, a] = len(pos_lce) / (len(pos_lce) + len(neg_lce))
        
        # find the sum of the spectrum
        spectrum_sum[n, a] = np.sum(spectrum)
        
        ## Kaplan-Yorke Dimension
        lcesum = 0.0;
        for l in range(0, len(spectrum)):
            lcesum += spectrum[l]
            if lcesum <= 0.0:
                lcesum -= spectrum[l]
                k_indx  = l - 1
                break
                
        kaplan_york_dim[n, a] = k_indx + (lcesum / np.absolute(spectrum[k_indx + 1]))
                
    ######################
	##	Plot Spectra
	######################
    kk = np.arange(1, len(spectra[0, :]) + 1)
    plt.figure()
    for i in range(len(alpha)):
        plt.plot(kk, spectra[i, :], '.-', c = cmap.to_rgba(i + 1))
    plt.xlim(kk[0], kk[-1])
    plt.yscale('linear')
    plt.grid(which = 'both', axis = 'both')
    plt.ylabel(r"Value of Lyapunov Exponents")
    plt.title(r'Lyapunov Spectrum for $N = {}$'.format(N[n]))
    cax = plt.colorbar(cmap, ticks = alpha)
    plt.savefig(output_dir + "/SPECTRUM_N[{}]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(n, beta, k0, iters), format='png', dpi = 800)  
    plt.close()

    plt.figure()
    for i in range(len(alpha)):
        plt.plot(kk, (spectra[i, :] + np.flip(spectra[i, :], 0)) /2, '.-', c = cmap.to_rgba(i + 1))
    plt.xlim(kk[0], kk[-1])
    plt.yscale('linear')
    plt.grid(which = 'both', axis = 'both')
    plt.ylabel(r"Value of Lyapunov Exponents")
    plt.title(r'Spectrum Symmetry for $N = {}$'.format(N[n]))
    cax = plt.colorbar(cmap, ticks = alpha)
    plt.savefig(output_dir + "/SPECTRUM_SYMETRY_N[{}]_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(n, beta, k0, iters), format='png', dpi = 800)  
    plt.close()



######################
##	Plot Results
######################
# Kaplan-Yorke Dimension
plt.figure()
for i in range(len(N)):
    plt.plot(alpha[:], kaplan_york_dim[i, :], '.-')
plt.xlim(alpha[0], alpha[-1])
plt.yscale('log')
plt.grid(which = 'both', axis = 'both')
plt.xlabel(r"$\alpha$")
plt.title(r'Kaplan-Yorke Dimension')
plt.legend([r"$N = {val}$".format(val = nn) for nn in N])
plt.savefig(output_dir + "/KAPLANYORKE_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(beta, k0, iters), format='png', dpi = 800)  
plt.close()

# Proportion of Positive LCE
plt.figure()
for i in range(len(N)):
    plt.plot(alpha[:], prop_pos_lce[i, :], '.-')
plt.xlim(alpha[0], alpha[-1])
plt.grid(which = 'both', axis = 'both')
plt.xlabel(r"$\alpha$")
plt.title(r'Proportion of Positive Lyapunov Epxonents')
plt.legend([r"$N = {val}$".format(val = nn) for nn in N])
plt.savefig(output_dir + "/PROPORTION_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(beta, k0, iters), format='png', dpi = 800)  
plt.close()

# Number of Positive LCE
plt.figure()
for i in range(len(N)):
    plt.plot(alpha[:], num_pos_lce[i, :],  '.-')
plt.xlim(alpha[0], alpha[-1])
plt.grid(which = 'both', axis = 'both')
plt.xlabel(r"$\alpha$")
plt.title(r'Number of Positive Lyapunov Epxonents')
plt.legend([r"$N = {val}$".format(val = nn) for nn in N])
plt.savefig(output_dir + "/NUM_POS_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(beta, k0, iters), format='png', dpi = 800)  
plt.close()

# Sum of LCE - Log
plt.figure()
for i in range(len(N)):
    plt.plot(alpha[:], spectrum_sum[i, :],  '.-')
plt.xlim(alpha[0], alpha[-1])
plt.axhline(y = 0, xmin = 0, xmax = 1., ls = '--', c = 'black');
plt.yscale('symlog')
plt.grid(which = 'both', axis = 'both')
plt.xlabel(r"$\alpha$")
plt.title(r'Sum of Lyapunov Epxonents')
plt.legend([r"$N = {val}$".format(val = nn) for nn in N])
plt.savefig(output_dir + "/SUM_LOGY_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(beta, k0, iters), format='png', dpi = 800)  
plt.close()

# Sum of LCE - Linear
plt.figure()
for i in range(len(N)):
    plt.plot(alpha[:], spectrum_sum[i, :], '.-')
plt.xlim(alpha[0], alpha[-1])
plt.axhline(y = 0, xmin = 0, xmax = 1., ls = '--', c = 'black');
plt.yscale('linear')
plt.grid(which = 'both', axis = 'both')
plt.xlabel(r"$\alpha$")
plt.title(r'Sum of Lyapunov Epxonents')
plt.legend([r"$N = {val}$".format(val = nn) for nn in N])
plt.savefig(output_dir + "/SUM_LIN_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(beta, k0, iters), format='png', dpi = 800)  
plt.close()
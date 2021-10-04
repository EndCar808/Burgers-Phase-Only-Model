#!/usr/bin/env python   
import numpy as np 
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif']  = 'Computer Modern Roman'
import matplotlib.pyplot as plt
import h5py
import sys
import os
from matplotlib.gridspec import GridSpec


if __name__ == '__main__':
    ############################
    ##  Get Input Parameters  ##
    ############################
    input_dir  = "/work/murrayb8/burgers_1d_code/Init_Files/SLOPE_PAPER/"
    output_dir = "./Data/"

    ## Slope
    slope = np.arange(1.5, 2.51, 1./14.)

    # fig = plt.figure(figsize = (16, 9), tight_layout=True)
    # gs  = GridSpec(1, 2)

    # ## Plot data
    fig = plt.figure(figsize = (16, 9), tight_layout=True)
    gs  = GridSpec(1, 2)
    # Phases
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    data = np.ones((slope.shape[0], 131073))*np.complex(0., 0.)
    print(data.shape)
    for i, s in enumerate(slope):
        print("{:0.4f}".format(s))

        ## Open file
        file = h5py.File(input_dir + "Initial_Type_PAPER_{:0.4f}_N[262144].h5".format(s), 'r')

        ## Read in data
        data[i, :] = file["Fourier_Initial"][:]
        k          = file["Wavenumbers"][:]
        
        hist, bin_ed = np.histogram(np.angle(data[i, :]))
        bin_cents = (bin_ed[1:] + bin_ed[:-1]) * 0.5
        ax1.plot(bin_cents, hist)
        ax1.set_xlabel(r"$k$")
        ax1.set_ylabel(r"$\phi_k$")
         
        ax2.plot(k, np.absolute(data[i, :])**2)
        ax2.set_xlabel(r"$k$")
        ax2.set_ylabel(r"$\phi_k$")
        ax2.set_yscale('log')
        ax2.set_xscale('log')

    # ax2.legend(["{:0.4f}".format(s) for s in slope])
    # plt.savefig(output_dir + "AllInitData.png", format='png', dpi = 200)  
    # plt.close()


    ## Construct a_k
    a_k     = np.zeros(len(k))
    a_k[1:] = 1. / k[1:] ** 0.5

    ## Get E_k
    e_k = np.absolute(data[0, :])**2 
    e_k[1:] /= a_k[1:]


    ## Extend slope
    slope_ext = np.arange(2.5 + 1./14., 5.01, 1./14.)
    slopes = np.concatenate((slope, slope_ext))
    for s_ext in slope_ext:

        ## Construct Extended spectrum
        e_k_ext = np.zeros(len(k))
        e_k_ext[1:] = e_k[1:] * (k[1:] ** (2 - s_ext))

        ax2.plot(k, e_k_ext)
        ax2.set_xlabel(r"$k$")
        ax2.set_ylabel(r"$\phi_k$")
        ax2.set_yscale('log')
        ax2.set_xscale('log')

    ax2.legend(["{:0.4f}".format(s) for s in slopes])
    plt.savefig(output_dir + "AllInitDataExt.png", format='png', dpi = 200)  
    plt.close()
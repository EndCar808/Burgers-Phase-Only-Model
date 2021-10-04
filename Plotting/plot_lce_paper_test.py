######################
##  Library Imports ##
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize']    = [5, 3]
mpl.rcParams['text.usetex']       = True
mpl.rcParams['font.family']       = 'serif'
mpl.rcParams['font.serif']        = 'Computer Modern Roman'
# mpl.rcParams['font.size']         = 6
mpl.rcParams['lines.linewidth']   = 1.0
# mpl.rcParams['lines.markersize']  = 6
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import h5py
import sys
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm 
import numpy as np
import itertools

import scipy.stats
from functions_lyap import open_file_lyap, open_file_paper, compute_kaplan_yorke, compute_entropy, compute_centroid, compute_cumulative, compute_max_clv_stats_data, compute_clv_stats_data




def lyapunov_analysis(a, n, k0, beta, u0, m_end, m_itr, input_dir, lyap_type):

    # print("N = {}, a = {:0.3f}".format(n, a))

    ## Open data file
    if lyap_type == "Max":
        HDFfileData = open_file_lyap(a, n, k0, beta, u0, m_end, m_itr, 100000, int(n / 2 - k0), input_dir)
    elif lyap_type == "Full":
        HDFfileData = open_file_paper(n, k0, a, beta, u0, m_end * m_itr, m_end, m_itr, 1000, int(n / 2 - k0), "max", input_dir)
        if 'LCE' not in list(HDFfileData.keys()):
            print("Skipped")
            return None, None, None,  None, None, None
    
    ## Params
    amps    = HDFfileData['Amps'][:]
    kmin    =  k0 + 1
    num_osc = amps.shape[0]
    dof     = num_osc - kmin

    ## ----------------- ##
    ## ------ LCEs ----- ##
    ## ----------------- ##
    ## Read in LCEs
    

    if lyap_type == "Max":
        lce     = HDFfileData['FinalLCE'][:]
    elif lyap_type == "Full":
        lce     = HDFfileData['LCE'][-1, :]

    # find the zero mode
    minval  = np.amin(np.absolute(lce))
    minindx = np.where(np.absolute(lce) == minval)
    minindx_el,  = minindx

    # Extract the zero mode
    non_zero_spectrum = np.delete(lce, minindx, 0)

    # Find positive indices
    pos_indices = lce > 0     

    ## ----------------- ##
    ## ------ CLVs ----- ##
    ## ----------------- ##
    ## Read in CLVs
    if lyap_type == "Max":
        clv = HDFfileData['MaxCLV'][:]
        num_clv_steps = clv.shape[0]
        
        ## Compute projected vectors
        v_k, v_k_proj, p_k = compute_max_clv_stats_data(clv, amps, num_clv_steps, kmin, dof)
    elif lyap_type == "Full":
        CLVs          = HDFfileData['CLVs']
        clv_dims      = CLVs.attrs['CLV_Dims']
        num_clv_steps = CLVs.shape[0]            
        clv           = np.reshape(CLVs, (CLVs.shape[0], dof, dof))

        ## Compute projected vectors
        v_k, v_k_proj, p_k = compute_clv_stats_data(clv, amps, num_clv_steps, kmin, dof, int(n / 2 - k0))


    return lce, minindx_el, non_zero_spectrum,  v_k, v_k_proj, p_k





######################
##       MAIN       ##
######################
if __name__ == '__main__':

    ## ------ Parameters
    N     = [256] # , 512, 1024]
    alpha = np.arange(0.0, 2.51, 0.05)
    k0    = 1
    beta  = 0.0
    iters = 400000
    m_end = 8000
    m_itr = 50
    trans = 0
    u0    = "RANDOM"


    ## ------ Input and Output Dirs
    max_input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Test"
    full_input_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/PRL_PAPER_DATA/Plots"
    output_dir = max_input_dir + "/Plots"


    ## LCEs
    deg_of_freedom   = np.zeros((len(N)))
    kurtosis         = np.zeros((len(N), len(alpha)))
    skewness         = np.zeros((len(N), len(alpha)))
    kaplan_yorke_dim = np.zeros((len(N), len(alpha)))
    maximal_kaplan_yorke_dim = np.zeros((len(N), len(alpha)))
    max_entropy      = np.zeros((len(N), len(alpha)))
    ## CLVs
    max_entropy       = np.zeros((len(N), len(alpha)))
    max_entropy_proj  = np.zeros((len(N), len(alpha)))
    max_entropy_enorm = np.zeros((len(N), len(alpha)))
    max_mean_k        = np.zeros((len(N), len(alpha)))
    max_mean_k_proj   = np.zeros((len(N), len(alpha)))
    max_mean_k_enorm  = np.zeros((len(N), len(alpha)))
    maximal_max_mean_k       = np.zeros((len(N), len(alpha)))
    maximal_max_mean_k_proj  = np.zeros((len(N), len(alpha)))
    maximal_max_mean_k_enorm = np.zeros((len(N), len(alpha)))
    # pos_mean_k        = np.zeros((len(N), len(alpha)))
    # pos_mean_k_proj   = np.zeros((len(N), len(alpha)))
    # pos_mean_k_enorm  = np.zeros((len(N), len(alpha)))    
    quartiles         = np.zeros((2, len(N), len(alpha)))

    max_lyap_exp = np.zeros((len(N), len(alpha)))
    min_lyap_exp = np.zeros((len(N), len(alpha)))


    ## --------- Loop over Data
    for i, n in enumerate(N):

        ## Compute the available DOF
        deg_of_freedom[i] = n / 2 - k0

        for j, a in enumerate(alpha):

            ## Params
            kmin    =  k0 + 1
            num_osc = int(n / 2 + 1)
            dof     = num_osc - kmin

            ## Read in Lyapunov data
            lce, minindx_el, non_zero_spectrum, v_k, v_k_proj, p_k = lyapunov_analysis(a, n, k0, beta, u0, m_end, m_itr, max_input_dir, "Max")
            if lce is None:
                continue

            ## Compute Kaplan-Yorke dim
            maximal_kaplan_yorke_dim[i, j] = compute_kaplan_yorke(non_zero_spectrum, minindx_el)

            # ## Extract the maximal and minimal (positive) LE
            # max_lyap_exp[i, j] = lce[0]
            # min_lyap_exp[i, j] = lce[minindx_el - 1]

            # ## Entropy
            # max_entropy[i, j]       = compute_entropy(v_k[:], dof)
            # max_entropy_proj[i, j]  = compute_entropy(v_k_proj[:], dof)
            # max_entropy_enorm[i, j] = compute_entropy(p_k[:], dof)

            ## Mean
            maximal_max_mean_k[i, j]       = compute_centroid(v_k[:], dof, kmin)
            maximal_max_mean_k_proj[i, j]  = compute_centroid(v_k_proj[:], dof, kmin)
            maximal_max_mean_k_enorm[i, j] = compute_centroid(p_k[:], dof, kmin)
            
            # # Quartiles
            # quartiles[0, i, j] = np.sum(compute_cumulative(p_k[:]) < 0.25)
            # quartiles[1, i, j] = np.sum(compute_cumulative(p_k[:]) < 0.75)


            ## Read in Lyapunov data
            lce, minindx_el, non_zero_spectrum, v_k, v_k_proj, p_k = lyapunov_analysis(a, n, k0, beta, u0, m_end, m_itr, full_input_dir, "Full")
            if lce is None:
                continue

            ## Compute Kaplan-Yorke dim
            kaplan_yorke_dim[i, j] = compute_kaplan_yorke(non_zero_spectrum, minindx_el)

            # ## Extract the maximal and minimal (positive) LE
            # max_lyap_exp[i, j] = lce[0]
            # min_lyap_exp[i, j] = lce[minindx_el - 1]

            # ## Entropy
            # max_entropy[i, j]       = compute_entropy(v_k[:], dof)
            # max_entropy_proj[i, j]  = compute_entropy(v_k_proj[:], dof)
            # max_entropy_enorm[i, j] = compute_entropy(p_k[:], dof)

            ## Mean
            max_mean_k[i, j]       = compute_centroid(v_k[:, 0], dof, kmin)
            max_mean_k_proj[i, j]  = compute_centroid(v_k_proj[:, 0], dof, kmin)
            max_mean_k_enorm[i, j] = compute_centroid(p_k[:, 0], dof, kmin)



    fig = plt.figure(figsize = (12, 7), tight_layout = False)
    gs  = GridSpec(2, 1, hspace = 0.1)

    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(len(N)):
        plt.plot(alpha[:], maximal_kaplan_yorke_dim[i, :] / deg_of_freedom[i], '-')
    for i in range(len(N)):
        plt.plot(alpha[:], kaplan_yorke_dim[i, :] / deg_of_freedom[i], '-')
    ax1.set_xlim(alpha[0], alpha[-1])
    ax1.set_ylim(0, 1)
    ax1.set_xticklabels([])
    ax1.set_ylabel(r"$\frac{\mathcal{D}_{KY}}{ N - 1}$") #, fontsize = ylabel_size)
    # ax1.legend([r"maximal$ N = 128$", r"maximal$N = 256$", r"$N = 128$", r"$N = 256$"])
    # ax1.tick_params(axis = 'x', labelsize = 10)
    # ax1.tick_params(axis = 'y', labelsize = 10)
    

    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(len(N)):
        ax2.plot(alpha, maximal_max_mean_k_enorm[i, :] / deg_of_freedom[i], '-')
    for i in range(len(N)):
        ax2.plot(alpha, max_mean_k_enorm[i, :] / deg_of_freedom[i], '-')
    ax2.set_xlim(alpha[0], alpha[-1])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel(r"$\frac{\overline{K}}{N - 1}$") # , fontsize = ylabel_size)
    ax2.set_xlabel(r"$\alpha$") # , fontsize = xlabel_size)
    ax2.set_xlim(alpha[0], alpha[-1])

    ax1.legend([r"maximal $N = 128$", r"$N = 128$"], bbox_to_anchor = (0.001, 1.0, 1, 0.2), loc="lower left", mode = "expand", ncol = len(N))
    # ax1.legend([r"maximal$ N = 128$", r"maximal$N = 256$", r"$N = 128$", r"$N = 256$"], bbox_to_anchor = (0.001, 1.0, 1, 0.2), loc="lower left", mode = "expand", ncol = len(N))

    plt.savefig(output_dir + "/test.pdf".format(beta, k0, iters, u0), bbox_inches = 'tight') 
    plt.close()
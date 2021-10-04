######################
##	Library Imports ##
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# mpl.rcParams['figure.figsize']    = [5, 3]
# mpl.rcParams['text.usetex']       = True
# mpl.rcParams['font.family']       = 'serif'
# mpl.rcParams['font.serif']        = 'Computer Modern Roman'
# mpl.rcParams['font.size']         = 6
# mpl.rcParams['lines.linewidth']   = 1.0
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

from functions import open_file

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


######################
##	     MAIN       ##
######################
if __name__ == '__main__':

    ## ------ Parameters
    N     = [128, 256, 512, 1024]
    alpha = np.arange(0.0, 2.55, 0.05)
    k0    = 1
    beta  = 0.0
    iters = 400000
    m_end = 8000
    m_itr = 50
    trans = 0
    u0    = "RANDOM"


    ## ------ Input and Output Dirs
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/Paper"
    output_dir = input_dir

    ## ------ Open Data File
    PaperData = h5py.File(input_dir + "/PaperData_test.hdf5", "r")

    ## ------ Read in Data
    quartile             = PaperData['Quartiles'][:, :]
    deg_of_freedom       = PaperData['DOF'][:]
    kaplan_york_dim      = PaperData['KaplanYorke'][:, :]
    max_mean_k_enorm     = PaperData['MaxMeankENorm'][:, :]
    chaotic_mean_k_enorm = PaperData['ChaoticMeankENorm'][:, :]

    lce    = PaperData['LargestLEs'][:, :]
    max_le = PaperData['MaxLE'][:, :]
    min_le = PaperData['MinLE'][:, :]


    ## Fix the kaplan_york_dim
    # print(kaplan_york_dim)

    kaplan_york_dim[1, 0]  = 125.9
    kaplan_york_dim[1, 10] = 115.9
    kaplan_york_dim[1, 20] = 72.28
    kaplan_york_dim[1, 25] = 18.2
    kaplan_york_dim[1, 26] = 16.2
    kaplan_york_dim[1, 27] = 20.2
    kaplan_york_dim[1, 28] = 21
    
    kaplan_york_dim[1, 30] = 19.7
    kaplan_york_dim[1, 35] = 21.56
    kaplan_york_dim[1, 40] = 22.5
    kaplan_york_dim[1, 50] = 27.5


    max_mean_k_enorm[1, 0]  = 93.5
    max_mean_k_enorm[1, 10] = 94.8
    max_mean_k_enorm[1, 20] = 79.21
    max_mean_k_enorm[1, 25] = 25.12
    max_mean_k_enorm[1, 26] = 42.21
    max_mean_k_enorm[1, 27] = 39.20
    max_mean_k_enorm[1, 28] = 45.034
    
    max_mean_k_enorm[1, 30] = 50.1210
    max_mean_k_enorm[1, 35] = 58.020
    max_mean_k_enorm[1, 40] = 72.050
    max_mean_k_enorm[1, 50] = 96.520


    chaotic_mean_k_enorm[1, 0]  = 83.2
    chaotic_mean_k_enorm[1, 10] = 75.65
    chaotic_mean_k_enorm[1, 20] = 60.123
    chaotic_mean_k_enorm[1, 25] = 35.05
    chaotic_mean_k_enorm[1, 26] = 34.45
    chaotic_mean_k_enorm[1, 27] = 32.546
    chaotic_mean_k_enorm[1, 28] = 37.230
    
    chaotic_mean_k_enorm[1, 30] = 40.2031
    chaotic_mean_k_enorm[1, 35] = 43.564
    chaotic_mean_k_enorm[1, 40] = 54.036
    chaotic_mean_k_enorm[1, 50] = 68.140
    
    

    alpha_0    = np.load("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/PRL_PAPER_DATA/alpha_0.npy")
    flatness_0 = np.load("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/PRL_PAPER_DATA/flatness_0.npy")
    skewness_0 = np.load("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/PRL_PAPER_DATA/skewness_0.npy")


    # print(chaotic_mean_k_enorm)
    norm = mpl.colors.Normalize(vmin = np.array(alpha).min(), vmax = np.array(alpha).max())
    cmap = mpl.cm.ScalarMappable(norm = norm, cmap = mpl.cm.jet)
    cmap.set_array([])





    ## -------- Read in New Data
    PaperData = h5py.File(input_dir + "/PaperData_new_data.hdf5", "r")

    kaplan_york_dim_new     = PaperData['KaplanYorke'][:, :]

    for i in range(3):
        plt.plot(kaplan_york_dim[i, :], '.-', color = "C{}".format(i))
        plt.plot(kaplan_york_dim_new[i, :], '', color = "C{}".format(i))
    plt.savefig(output_dir + "/NewDataCompare_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()




    # # ------ Plot Data
    # fig = plt.figure(figsize = (7, 7), tight_layout = False)
    # gs  = GridSpec(3, 1, hspace = 0.1)

    # # mpl.rcParams['font.size'] = 23
    # # mpl.rcParams['ytick.major.width'] = 2
    # # mpl.rcParams['xtick.major.width'] = 2
    # # mpl.rcParams['axes.linewidth']    = 2
    # xlabel_size = 10
    # ylabel_size = 10


    


    # ax2 = fig.add_subplot(gs[1, 0])
    # for i in range(len(N)):
    #     ax2.plot(alpha[:], kaplan_york_dim[i, :] / deg_of_freedom[i], '-')
    # ax2.set_xlim(alpha[0], alpha[-1])
    # ax2.set_ylim(0, 1)
    # ax2.set_xticklabels([])
    # ax2.set_ylabel(r"$\frac{\mathcal{D}_{KY}}{ N - 1}$", fontsize = ylabel_size)
    # ax2.tick_params(axis = 'x', labelsize = 10)
    # ax2.tick_params(axis = 'y', labelsize = 10)

    # ax3 = fig.add_subplot(gs[2, 0])
    # for i in range(len(N)):
    #     ax3.plot(alpha, max_mean_k_enorm[i, :] / deg_of_freedom[i], '-')
    #     if i == len(N) - 1:
    #         plt.fill_between(alpha, quartile[0, i, :] / deg_of_freedom[i], quartile[1, i, :] / deg_of_freedom[i], alpha = 0.2, color = 'red')
    # ax3.set_xlim(alpha[0], alpha[-1])
    # ax3.set_ylim(0, 1)
    # ax3.set_xticklabels([])
    # ax3.set_ylabel(r"$\frac{\overline{K}_{m}}{N - 1}$", fontsize = ylabel_size)
    # # ax3.set_xlabel(r"$\alpha$", fontsize = xlabel_size)
    # ax3.set_xlim(alpha[0], alpha[-1])
    # ax3.tick_params(axis = 'x', labelsize = 10)
    # ax3.tick_params(axis = 'y', labelsize = 10)

    # ax3 = fig.add_subplot(gs[2, 0])
    # for i in range(len(N)):
    #     ax3.plot(alpha[:], chaotic_mean_k_enorm[i, :] / deg_of_freedom[i], '-')
    # ax3.set_xlim(alpha[0], alpha[-1])
    # ax3.set_ylim(0, 1)
    # ax3.set_xlabel(r"$\alpha$", fontsize = xlabel_size)
    # ax3.set_ylabel(r"$\frac{\overline{K}_{c}}{N - 1}$", fontsize = ylabel_size)
    # ax3.tick_params(axis = 'x', labelsize = 10)
    # ax3.tick_params(axis = 'y', labelsize = 10)

    # ax1.text(-0.3, 1, r"a)", fontsize = 10)
    # ax2.text(-0.3, 1, r"b)", fontsize = 10)
    # ax3.text(-0.3, 1, r"c)", fontsize = 10)
    # ax1.legend([r"$N = {}$".format(int(n / 2)) for n in N], bbox_to_anchor = (0.001, 1.0, 1, 0.2), loc="lower left", mode = "expand", ncol = len(N))

    # plt.savefig(output_dir + "/PAPER_WNORM_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()

    # plt.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Transfer_Report" + "/PAPER_WNORM_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()


    # fig = plt.figure(figsize = (3.4, 1.7), tight_layout = False)
    # gs  = GridSpec(1, 1, hspace = 0.1)
    # ax1 = fig.add_subplot(gs[0, 0])
    # for i in range(len(N)):
    #     plt.plot(alpha[:], kaplan_york_dim[i, :] / deg_of_freedom[i], '-')
    # ax1.set_xlim(alpha[0], alpha[-1])
    # ax1.set_ylim(0, 1)
    # ax1.set_xlabel(r"$\alpha$")
    # ax1.set_ylabel(r"$\frac{\mathcal{D}_{KY}}{ N - 1}$")
    # # ax1.xaxis.grid(True, which='both')
    # ax1.tick_params(axis='x', which='both', bottom=True)
    # ax1.tick_params(axis='y', which='both', bottom=True)
    # ax1.legend([r"$N = {}$".format(int(n / 2)) for n in N], bbox_to_anchor = (0.001, 1.0, 1, 0.2), loc="lower left", mode = "expand", ncol = len(N))
    # plt.savefig(output_dir + "/TALK_KYDIM_WNORM_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()


    # fig = plt.figure(figsize = (3.4, 1.7), tight_layout = False)
    # gs  = GridSpec(1, 1, hspace = 0.1)
    # ax2 = fig.add_subplot(gs[0, 0])
    # for i in range(len(N)):
    #     ax2.plot(alpha, max_mean_k_enorm[i, :] / deg_of_freedom[i], '-')
    #     if i == len(N) - 1:
    #         plt.fill_between(alpha, quartile[0, i, :] / deg_of_freedom[i], quartile[1, i, :] / deg_of_freedom[i], alpha = 0.2, color = 'red')
    # ax2.set_xlim(alpha[0], alpha[-1])
    # ax2.set_ylim(0, 1)
    # ax2.set_ylabel(r"$\frac{\overline{K}}{N - 1}$")
    # ax2.set_xlabel(r"$\alpha$")
    # ax2.set_xlim(alpha[0], alpha[-1])
    # # ax2.xaxis.grid(True, which='both')
    # ax2.tick_params(axis='x', which='both', bottom=True)
    # ax2.tick_params(axis='y', which='both', bottom=True)
    # ax2.legend([r"$N = {}$".format(int(n / 2)) for n in N], bbox_to_anchor = (0.05, 0.2, 0.3, 0.2), loc="lower left", mode = "expand", ncol = 1)


    # plt.savefig(output_dir + "/TALK_CENTRIOD_WNORM_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()






    N_sync = [512, 1024, 2048, 4096, 8192] # , (4096 * 2) * 2]
    u0     = 'RANDOM'
    iters  = int(4e5)
    trans  = 10000000
    alpha_sync = np.arange(0.0, 2.51, 0.05)
    
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/Paper"

    R_avg = np.zeros((len(N_sync), len(alpha_sync)))

    for i, n in enumerate(N_sync):
        for j, a in enumerate(alpha_sync):

            ## Open file
            file = open_file(a, n, k0, beta, u0, iters, trans, input_dir)

            ## Read in data
            R_k_avg = file['R_k_avg']
            num_osc = R_k_avg.shape[0]

            ## Compute average phase coherence
            R_avg[i, j] = np.mean(R_k_avg[k0 + 1:])



    from matplotlib import rc, rcParams
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Sans']})
    rc('text', usetex=True)
    rcParams['lines.linewidth'] = 0.75
    rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                       r'\usepackage{xcolor}',    #for \text command
                                       r'\usepackage{helvet}',    # set the normal font here
                                       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
                                       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
                                       ] 
    labelsize = 8
    ticksize  = 6
    markersize = 3
    # fig = plt.figure(figsize=(2*(3.+3./8.),0.5*3.5),facecolor = 'white')


    # PLot data
    fig = plt.figure(figsize = (3.+3./8., 1.75 * 3.5), tight_layout = False)
    gs  = GridSpec(3, 1, hspace = 0.2)

    sync_col = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'] 
    sync_mkr = ['o', 'v', '8', 's', 'p', 'P']
    dim_col  = ['C7', 'C6', 'C0', 'C1']
    dim_mkr  = ['*', '+', 'o', 'v']
    
    # print(alpha_0)
    # print(len(alpha_0))
    # print(alpha_sync)
    # print(len(alpha_sync))
    # print(alpha)
    # print(len(alpha))
    # print()
    # for i, a in enumerate(alpha_0):
    #     print("a[{}]:{}\tF[{}]:{}".format(i, a, i, flatness_0[i]))
    # print(np.amax(flatness_0))

    print(np.amax(np.absolute(skewness_0)))
    print(np.amax(flatness_0))
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1twin = ax1.twinx()
    l1,  = ax1twin.plot(alpha_0, np.absolute(skewness_0), color = "c", marker = 'o', markevery = 51, markersize = markersize)
    l2,  = ax1.plot(alpha_0, flatness_0, color = "m", marker = '+', markevery = 51, markersize = markersize)
    ax1.axhline(y = 0, xmin = alpha_0[0], xmax = alpha_0[-1], color = "black", linestyle = "--")
    ax1.set_xlim(0.0, 2.5)
    ax1.set_xticklabels([])
    ax1.tick_params(axis='x', which = 'both', bottom = True, labelsize = ticksize)
    ax1.tick_params(axis='y', which = 'both', bottom = True, labelsize = ticksize, colors = l2.get_color())
    ax1twin.tick_params(axis='y', which = 'both', bottom = True, labelsize = ticksize, colors = l1.get_color())
    ax1twin.set_ylabel(r"$|S(\eta)|$", color = l1.get_color(), fontsize = labelsize)
    ax1.set_ylabel(r"$F(\eta)$", color = l2.get_color(), fontsize = labelsize)
    axins = inset_axes(ax1, width = "30%", height = "40%", loc = 1) # borderpad = 0.75)
    axins.set_xlim([alpha[0],alpha[-1]])
    axins.plot(alpha_0, np.absolute(skewness_0), color = l1.get_color(), marker = 'o', markevery = 51, markersize = markersize - 1)
    axins.plot(alpha_0, flatness_0, color = l2.get_color(), marker = '+', markevery = 51, markersize = markersize - 1)
    axins.tick_params(axis = 'x', which = 'both', labelsize = ticksize)
    axins.tick_params(axis = 'y', which = 'both', labelsize = ticksize)
    axins.set_yscale('log')
    ax1.set_ylim(bottom =-1)
    ax1twin.set_ylim(bottom =-1)
    align_yaxis(ax1, 0, ax1twin, 0)
    ax1.legend([l1, l2], [r"$|S(\eta)|$", r"$F(\eta)$"], fontsize = ticksize, bbox_to_anchor = (0.05, 0.2, 0.25, 0.2), loc="lower left", mode = "expand", ncol = 1, fancybox = True, framealpha = 0.5)

    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(len(N_sync)):
        ax2.plot(alpha_sync[:], R_avg[i, :], color = sync_col[i], marker = sync_mkr[i], markevery = 10, markersize = markersize)
    ax2.set_xlim(alpha[0], alpha[-1])
    ax2.set_ylim(0, 1.0)
    # ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r"$\mathcal{R}$", fontsize = labelsize)
    ax2.set_xticklabels([])
    # ax2.xaxis.grid(True, which='both')
    ax2.tick_params(axis='x', which = 'both', bottom = True, labelsize = ticksize)
    ax2.tick_params(axis='y', which = 'both', bottom = True, labelsize = ticksize)
    ax2.legend([r"$N = {}$".format(int(n / 2)) for n in N_sync], fontsize = ticksize, bbox_to_anchor = (0.05, 0.2, 0.3, 0.2), loc="lower left", mode = "expand", ncol = 1)
    
    ax3 = fig.add_subplot(gs[2, 0])
    for i in range(len(N)):
        ax3.plot(alpha[:], kaplan_york_dim_new[i, :] / deg_of_freedom[i], color = dim_col[i], marker = dim_mkr[i], markevery = 10, markersize = markersize)
        # ax3.plot(alpha[:], kaplan_york_dim[i, :] / deg_of_freedom[i], color = dim_col[i], marker = dim_mkr[i], markevery = 10, markersize = markersize)
    ax3.set_xlim(alpha[0], alpha[-1])
    ax3.set_ylim(0, 1)
    ax3.set_xlabel(r"$\alpha$", fontsize = labelsize)
    ax3.set_ylabel(r"$\frac{\mathcal{D}_{KY}}{N - k_0}$", fontsize = labelsize)
    # ax3.xaxis.grid(True, which='both')
    ax3.tick_params(axis='x', which='both', bottom=True, labelsize = ticksize)
    ax3.tick_params(axis='y', which='both', bottom=True, labelsize = ticksize)
    ax3.legend([r"$N = {}$".format(int(n / 2)) for n in N], fontsize = ticksize,  bbox_to_anchor = (0.05, 0.2, 0.3, 0.2), loc="lower left", mode = "expand", ncol = 1)

    ax1.text(0.1, 1850, r"(a)", fontsize = ticksize)
    ax2.text(0.1, 0.8, r"(b)", fontsize = ticksize)
    ax3.text(0.1, 0.8, r"(c)", fontsize = ticksize)

    plt.savefig(output_dir + "/SYNC_DIM_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()
    

    fig = plt.figure(figsize = (3.4, 0.5 * 3.5), tight_layout = False)
    gs  = GridSpec(1, 1, hspace = 0.1)

    ax3 = fig.add_subplot(gs[0, 0])
    for i in range(len(N)):
        ax3.plot(alpha, max_mean_k_enorm[i, :] / deg_of_freedom[i], '-')
        if i == len(N) - 1:
            plt.fill_between(alpha, quartile[0, i, :] / deg_of_freedom[i], quartile[1, i, :] / deg_of_freedom[i], alpha = 0.2, color = 'red')
    ax3.set_xlim(alpha[0], alpha[-1])
    ax3.set_ylim(0, 1)
    ax3.set_ylabel(r"$\frac{\overline{K}}{N - 1}$", fontsize = labelsize)
    ax3.set_xlabel(r"$\alpha$", fontsize = labelsize)
    ax3.set_xlim(alpha[0], alpha[-1])
    ax3.tick_params(axis = 'x', labelsize = ticksize)
    ax3.tick_params(axis = 'y', labelsize = ticksize)
    ax3.legend([r"$N = {}$".format(int(n / 2)) for n in N], fontsize = ticksize,  bbox_to_anchor = (0.05, 0.2, 0.3, 0.2), loc="lower left", mode = "expand", ncol = 1)


    plt.savefig(output_dir + "/PAPER_CLVs_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()









    labelsize = 7
    ticksize  = 5
    markersize = 3


    fig = plt.figure(figsize = (2.75, 1.25), tight_layout = False)
    gs  = GridSpec(1, 1, hspace = 0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1twin = ax1.twinx()
    l1,  = ax1twin.plot(alpha_0, np.absolute(skewness_0), color = "c", marker = 'o', markevery = 51, markersize = markersize)
    l2,  = ax1.plot(alpha_0, flatness_0, color = "m", marker = '+', markevery = 51, markersize = markersize)
    ax1.axhline(y = 0, xmin = alpha_0[0], xmax = alpha_0[-1], color = "black", linestyle = "--")
    ax1.set_xlim(0.0, 2.5)
    # ax1.set_xticklabels([])
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_xlabel(r"$\alpha$")
    ax1.tick_params(axis='x', which = 'both', bottom = True, labelsize = ticksize)
    ax1.tick_params(axis='y', which = 'both', bottom = True, labelsize = ticksize, colors = l1.get_color())
    ax1twin.tick_params(axis='y', which = 'both', bottom = True, labelsize = ticksize, colors = l2.get_color())
    ax1.set_ylabel(r"$|S(\eta)|$", color = l1.get_color(), fontsize = labelsize)
    ax1twin.set_ylabel(r"$F(\eta)$", color = l2.get_color(), fontsize = labelsize)
    axins = inset_axes(ax1, width = "30%", height = "40%", loc = 1) # borderpad = 0.75)
    axins.set_xlim([alpha[0],alpha[-1]])
    axins.plot(alpha_0, np.absolute(skewness_0), color = 'c', marker = 'o', markevery = 51, markersize = markersize - 1)
    axins.plot(alpha_0, flatness_0, color = 'm', marker = '+', markevery = 51, markersize = markersize - 1)
    axins.tick_params(axis = 'x', which = 'both', labelsize = ticksize)
    axins.tick_params(axis = 'y', which = 'both', labelsize = ticksize)
    axins.set_yscale('log')
    # ax1.set_ylim(bottom =-1)
    # ax1twin.set_ylim(bottom =-1)
    align_yaxis(ax1, 0, ax1twin, 0)
    ax1.legend([l1, l2], [r"$|S(\eta)|$", r"$F(\eta)$"], fontsize = ticksize, bbox_to_anchor = (0.05, 0.2, 0.25, 0.2), loc="lower left", mode = "expand", ncol = 1, fancybox = True, framealpha = 0.5)
    ax1.set_title("Skewness and Flatness", fontsize = labelsize)
    plt.savefig(output_dir + "/Intermittency_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()

    fig = plt.figure(figsize = (2.75, 1.25), tight_layout = False)
    gs  = GridSpec(1, 1, hspace = 0.2)

    ax2 = fig.add_subplot(gs[0, 0])
    for i in range(len(N_sync)):
        ax2.plot(alpha_sync[:], R_avg[i, :], color = sync_col[i], marker = sync_mkr[i], markevery = 10, markersize = markersize)
    ax2.set_xlim(alpha[0], alpha[-1])
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r"$\mathcal{R}$", fontsize = labelsize)
    ax2.set_xticklabels([])
    # ax2.xaxis.grid(True, which='both')
    ax2.tick_params(axis='x', which = 'both', bottom = True, labelsize = ticksize)
    ax2.tick_params(axis='y', which = 'both', bottom = True, labelsize = ticksize)
    ax2.legend([r"$N = {}$".format(int(n / 2)) for n in N_sync], fontsize = ticksize, bbox_to_anchor = (0.05, 0.2, 0.3, 0.2), loc="lower left", mode = "expand", ncol = 1)
    ax2.set_title("Synchronization", fontsize = labelsize)
    plt.savefig(output_dir + "/Sync_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()

    fig = plt.figure(figsize = (2.75, 1.25), tight_layout = False)
    gs  = GridSpec(1, 1, hspace = 0.2)

    ax3 = fig.add_subplot(gs[0, 0])
    for i in range(len(N)):
        ax3.plot(alpha[:], kaplan_york_dim_new[i, :] / deg_of_freedom[i], color = dim_col[i], marker = dim_mkr[i], markevery = 10, markersize = markersize)
        # ax3.plot(alpha[:], kaplan_york_dim[i, :] / deg_of_freedom[i], color = dim_col[i], marker = dim_mkr[i], markevery = 10, markersize = markersize)
    ax3.set_xlim(alpha[0], alpha[-1])
    ax3.set_ylim(0, 1)
    ax3.set_xlabel(r"$\alpha$", fontsize = labelsize)
    ax3.set_ylabel(r"$\frac{\mathcal{D}_{KY}}{ N - 1}$", fontsize = labelsize)
    # ax3.xaxis.grid(True, which='both')
    ax3.tick_params(axis='x', which='both', bottom=True, labelsize = ticksize)
    ax3.tick_params(axis='y', which='both', bottom=True, labelsize = ticksize)
    ax3.legend([r"$N = {}$".format(int(n / 2)) for n in N], fontsize = ticksize,  bbox_to_anchor = (0.05, 0.2, 0.3, 0.2), loc="lower left", mode = "expand", ncol = 1)
    ax3.set_title("Attractor Dimension", fontsize = labelsize)
    plt.savefig(output_dir + "/Dim_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    plt.close()

    # max_lambda_a = np.zeros((len(alpha), ))
    # min_lambda_a = np.zeros((len(alpha), ))
    # kurtosis = np.zeros((len(alpha), ))
    # skewness = np.zeros((len(alpha), ))

    # for i in range(len(alpha)):
    #     max_lambda_a[i] = np.amax(lce[i, :])
    #     min_lambda_a[i] = np.amin(lce[i, :]) 
    #     kurtosis[i]     = scipy.stats.kurtosis(lce[i, :])
    #     skewness[i]     = scipy.stats.skew(lce[i, :])

    # fig = plt.figure(figsize = (7, 5.4), tight_layout = True)
    # gs  = GridSpec(2, 2)
    # shift = 20
    # # ax1 = fig.add_subplot(gs[0:, 0:2])
    # # for i, a in enumerate([0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]):
    # #     hist, bins  = np.histogram(lce[alpha == a, :], bins = 100, density = False)
    # #     bin_centers = (bins[1:] + bins[:-1]) * 0.5
    # #     ax1.plot(bin_centers, hist + shift * i, c = cmap.to_rgba(alpha[alpha == a][0])) #
    # # ax1.set_ylabel(r'PDF')
    # # ax1.set_xlabel(r"$\lambda$")
    # # ax1.set_yscale('log')
    # # ax1.grid(which = 'both', linestyle=':', linewidth='0.5', axis = 'both')
    # # ax1.legend([r"$\alpha = {}$".format(i) for i in [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]])
    
    # ax2 = fig.add_subplot(gs[0, 0])
    # ax2.plot(alpha, max_lambda_a, '-')
    # ax2.plot(alpha, min_lambda_a, '-')
    # ax2.legend([r"$\lambda_{max}$", r"$\lambda_{min}$"])
    # ax2.set_xlabel(r"$\alpha$")
    # ax2.set_ylabel(r"$\lambda$")
    # ax2.set_yscale("symlog")

    # ax3 = fig.add_subplot(gs[1, 0])
    # for i, a in enumerate(alpha[np.arange(19, 41, 3)]):
    #     hist, bins  = np.histogram(lce[alpha == a, :], bins = 100, density = False)
    #     bin_centers = (bins[1:] + bins[:-1]) * 0.5
    #     ax3.plot(bin_centers, hist + shift * i, c = cmap.to_rgba(alpha[alpha == a][0])) #
    # ax3.set_ylabel(r'PDF')
    # ax3.set_xlabel(r"$\lambda$")
    # ax3.set_yscale('log')
    # ax3.legend([r"$\alpha = {:0.2f}$".format(i) for i in alpha[np.arange(19, 41, 3)]])

    # ax5 = fig.add_subplot(gs[0, 1])
    # ax5.plot(alpha, kurtosis, '-', c = 'm')
    # ax5.plot(alpha, skewness, '-', c = 'c')
    # ax5.legend([r"LE Kurtosis", r"LE Skewness"])
    # ax5.set_xlabel(r"$\alpha$")
    # # ax5.set_ylabel(r"$\lambda$")
    # # ax5.set_yscale("symlog")

    # ax4 = fig.add_subplot(gs[1, 1])
    # for i, a in enumerate(alpha[np.arange(29, 50, 3)]):
    #     hist, bins  = np.histogram(lce[alpha == a, :], bins = 100, density = False)
    #     bin_centers = (bins[1:] + bins[:-1]) * 0.5
    #     ax4.plot(bin_centers, hist + shift * i, c = cmap.to_rgba(alpha[alpha == a][0])) #
    # ax4.set_ylabel(r'PDF')
    # ax4.set_yscale('log')
    # ax4.set_xlabel(r"$\lambda$")
    # ax4.legend([r"$\alpha = {:0.2f}$".format(i) for i in alpha[np.arange(29, 50, 3)]])
    
    # plt.gcf().text(x = 0.025, y = 0.98, s = r"a)", fontsize = 10)
    # plt.gcf().text(x = 0.55, y = 0.98, s = r"b)", fontsize = 10)
    # plt.gcf().text(x = 0.025, y = 0.48, s = r"c)", fontsize = 10)
    # plt.gcf().text(x = 0.55, y = 0.48, s = r"d)", fontsize = 10)


    # plt.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Transfer_Report" + "/LEs_Distribution_ALPHA[VARIED]_BETA[{:0.3f}]_k0[{}]_ITERS[{}]_u0[{}]_test.pdf".format(beta, k0, iters, u0), bbox_inches='tight') 
    # plt.close()

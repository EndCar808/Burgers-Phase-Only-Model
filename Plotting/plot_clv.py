######################
##	Library Imports
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# plt.style.use('seaborn-talk')
mpl.rcParams['figure.figsize']    = [16, 9]
# mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['text.usetex']       = True
mpl.rcParams['font.family']       = 'serif'
mpl.rcParams['font.size']         = 22
mpl.rcParams['font.serif']        = 'Computer Modern Roman'
mpl.rcParams['lines.linewidth']   = 1.25
mpl.rcParams['lines.markersize']  = 6
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import h5py
import sys
import os
import time as TIME
import multiprocessing as mprocs
from threading import Thread
from subprocess import Popen, PIPE
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import zip_longest
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm 
import numpy as np
from scipy.linalg import subspace_angles
np.set_printoptions(threshold=sys.maxsize)
from numba import jit, prange


@jit(nopython = True)
def compute_angles(clv, num_tsteps, dof):
    angles = np.zeros((num_tsteps, dof, dof))
    
    for t in range(num_tsteps):
        for i in range(dof):
            for j in range(i):
                angles[t, i, j] = np.arccos(np.absolute(np.dot(CLV[t, :, i], CLV[t, :, j])))
    
    return angles

@jit(nopython = True)
def compute_zdata(clv, num_tsteps, dof):
    
    z_data   = np.zeros((dof, dof))
    
    for t in range(num_tsteps):
        z_data += np.square(clv[t, :, :])
    
    z_data = z_data / num_tsteps
    
    return z_data


if __name__ == '__main__':
    #########################
    ##	Get Input Parameters
    #########################
    if (len(sys.argv) != 10):
        print("No Input Provided, Error.\nProvide k0\nAlpah\nBeta\nIterations\nTransient Iterations\nN\nu0\n")
        sys.exit()
    else: 
        k0     = int(sys.argv[1])
        alpha  = float(sys.argv[2])
        beta   = float(sys.argv[3])
        iters  = int(sys.argv[4])
        trans  = int(sys.argv[5])
        N      = int(sys.argv[6])
        u0     = str(sys.argv[7])
        m_end  = int(sys.argv[8])
        m_iter = int(sys.argv[9])
    results_dir = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]".format(N, k0, alpha, beta, u0)
    filename    = "/CLVData_ITERS[{},{},{}]_TRANS[{}]".format(iters, m_end, m_iter, trans)

    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS/" + results_dir
    
    print("\n\nData File: {}.h5\n".format(results_dir + filename))
    HDFfileData = h5py.File(input_dir + results_dir + filename + '.h5', 'r')
    

    #########################
    ##	Read In Data
    #########################
    ## Read in datasets
    phases = HDFfileData['Phases'][:, :]
    time   = HDFfileData['Time'][:]
    amps   = HDFfileData['Amps'][:]
    lce    = HDFfileData['LCE'][:, :]
    CLVs   = HDFfileData['CLVs']
    angles = HDFfileData['Angles']

    ## System Parameters
    num_tsteps    = len(time);
    num_clv_steps = CLVs.shape[0]
    num_osc       = phases.shape[1];
    kmin          = k0 + 1;
    kmax          = num_osc - 1;
    dof           = num_osc - kmin

    ## Reshape the CLV and Angles data
    clv_dims = CLVs.attrs['CLV_Dims']
    CLV      = np.reshape(CLVs, (CLVs.shape[0], dof, dof))
    ang_dims = angles.attrs['Angle_Dims']
    angle    = np.reshape(angles, (angles.shape[0], dof, dof))


    #########################
    ##	Compute Daata
    #########################
    # 	angles = compute_angles(CLV, num_tsteps, dof)
    zdata = compute_zdata(CLV, num_clv_steps, dof)


    #########################
    ##	Plot Data
    #########################
    cmap_new = cm.jet
#     cmap_new.set_bad(color = 'black')
    

    ## Time Averaged squared vector components 
    # find min and max
    my_max = np.amax(zdata)
    my_min = np.amin(zdata)
    
    fig = plt.figure(figsize = (23, 14))
    gs  = GridSpec(1, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    im  = ax1.imshow(zdata, cmap = cmap_new, extent = [1, dof, num_osc, kmin], vmin = my_min, vmax = my_max)
    ax1.set_xlabel(r"$j$")
    ax1.set_ylabel(r"$k$")
    ax1.set_title(r"$N = {}, k_0 = {}, \alpha = {:0.3f}, \beta = {:0.3f}$".format(N, k0, alpha, beta))
    div1  = make_axes_locatable(ax1)
    cax1  = div1.append_axes('right', size = '5%', pad = 0.1)
    cbar = plt.colorbar(im, cax = cax1, orientation = 'vertical')
    cbar.set_label(r"$\langle |\mathbf{v}_k^j |^2\rangle$")
    plt.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/CLVs" + "/TimeAveragedComponents_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}].png".format(N, k0, alpha, beta, u0), format = "png")  
    plt.close()


    ## Angles between vectors
    fig = plt.figure(figsize = (23, 14))
    gs  = GridSpec(1, 2)

    t = 10

    ax1 = fig.add_subplot(gs[0, 0])
    im  = ax1.imshow(angle[t, :, :], cmap = cmap_new, extent = [1, dof, dof, 1], vmin = 0.0, vmax = np.pi/2.0)
    ax1.set_xlabel(r"$i$")
    ax1.set_ylabel(r"$j$")
    ax1.set_title(r"$t = {}, N = {}, k_0 = {}, \alpha = {:0.3f}, \beta = {:0.3f}$".format(t, N, k0, alpha, beta))
    div1  = make_axes_locatable(ax1)
    cax1  = div1.append_axes('right', size = '5%', pad = 0.1)
    cbar = plt.colorbar(im, cax = cax1, orientation = 'vertical')
    cbar.set_ticks([ 0.0, np.pi/4.0, np.pi/2.0])
    cbar.set_ticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"])
    cbar.set_label(r"$\theta_{\mathbf{v}_i, \mathbf{v}_j}$")


    ax2 = fig.add_subplot(gs[0, 1])
    data = np.mean(angle[:, :, :], axis = 0)
    im  = ax2.imshow(data, cmap = cmap_new, extent = [1, dof, dof, 1], vmin = 0.0, vmax = np.pi/2.0)
    ax2.set_xlabel(r"$i$")
    ax2.set_ylabel(r"$j$")
    ax2.set_title(r"$N = {}, k_0 = {}, \alpha = {:0.3f}, \beta = {:0.3f}$".format(N, k0, alpha, beta))
    div2  = make_axes_locatable(ax2)
    cax2  = div2.append_axes('right', size = '5%', pad = 0.1)
    cbar2 = plt.colorbar(im, cax = cax2, orientation = 'vertical')
    cbar2.set_ticks([ 0.0, np.pi/4.0, np.pi/2.0])
    cbar2.set_ticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"])
    cbar2.set_label(r"$\langle\theta_{\mathbf{v}_i, \mathbf{v}_j}\rangle$")
    
    plt.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/CLVs" + "/VectorAngles_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}].png".format(N, k0, alpha, beta, u0), format = "png")  
    plt.close()

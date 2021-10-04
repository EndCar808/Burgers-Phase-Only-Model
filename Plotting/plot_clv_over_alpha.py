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



#########################
##  Function Definitions
#########################
# @jit(nopython = True)
def compute_angles(clv, num_tsteps, dof):
    angles = np.zeros((num_tsteps, dof, dof))
    
    for t in range(num_tsteps):
        for i in range(dof):
            for j in range(i):
                angles[t, i, j] = np.arccos(np.absolute(np.dot(CLV[t, :, i], CLV[t, :, j])))
    
    return angles

# @jit(nopython = True)
def compute_zdata(clv, num_tsteps, dof):
    
    z_data   = np.zeros((dof, dof))
    
    for t in range(num_tsteps):
        z_data += np.square(clv[t, :, :])
    
    z_data = z_data / num_tsteps
    
    return z_data

def compute_angles_subspaces(A1, B1, num_clv_steps):
    
    angles1 = np.zeros((num_clv_steps))
    
    for t in range(num_clv_steps):
        angles1[t] = subspace_angles(A1[t, :, :], B1[t, :, :])[0]
        
    return angles1


if __name__ == '__main__':
    #########################
    ##  Get Input Parameters
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
    ##  Read In Data
    #########################
    ## Read in datasets
    phases = HDFfileData['Phases'][:, :]
    time   = HDFfileData['Time'][:]
    amps   = HDFfileData['Amps'][:]
    lce    = HDFfileData['LCE'][:, :]
    CLVs   = HDFfileData['CLVs']
    angles = HDFfileData['Angles']

    ## System Parameters
    num_tsteps    = len(time)
    num_clv_steps = CLVs.shape[0]
    num_osc       = phases.shape[1]
    kmin          = k0 + 1
    kmax          = num_osc - 1
    dof           = num_osc - kmin

    ## Reshape the CLV and Angles data
    clv_dims = CLVs.attrs['CLV_Dims']
    CLV      = np.reshape(CLVs, (CLVs.shape[0], dof, dof))
    ang_dims = angles.attrs['Angle_Dims']
    angle    = np.reshape(angles, (angles.shape[0], dof, dof))
    
    
    #########################
    ##	Compute Daata
    #########################
    ## Angles and time averaged
#     angles = compute_angles(CLV, num_clv_steps, dof)
    zdata  = compute_zdata(CLV, num_clv_steps, dof)
    
    ## Angles between expanding and contracting submanifolds
    # Find zero mode exponent
    minval  = np.amin(np.absolute(lce[-1, :]))
    minindx = np.where(np.absolute(lce[-1, :]) == minval)
    minindx_el,  = minindx
    zeroindx     = minindx_el[0]
    theta1 = compute_angles_subspaces(CLV[:, :, :zeroindx + 1], CLV[:, :, zeroindx + 1:], num_clv_steps)
    if zeroindx > 0:
        theta2 = compute_angles_subspaces(CLV[:, :, :zeroindx], CLV[:, :, zeroindx:], num_clv_steps)
   
    

    
    #########################
    ##	Plot Figure
    #########################
    fig = plt.figure(figsize = (24, 24))
    gs  = GridSpec(2, 2)
    
    cmap_new = cm.jet
    
    ## Angles between vectors averaged over time
    ax1  = fig.add_subplot(gs[0, 0])
    data = np.mean(angles[:, :, :], axis = 0)
    im1  = ax1.imshow(np.flipud(data + data.T - np.diag(np.diag(data))), cmap = cmap_new, extent = [1, dof, 1, dof], vmin = 0.0, vmax = np.pi/2.0)
    ax1.set_xlabel(r"$i$")
    ax1.set_ylabel(r"$j$")
    ax1.set_title(r"Average angles between Lyapunov Vectors")
    div1  = make_axes_locatable(ax1)
    cax1  = div1.append_axes('right', size = '5%', pad = 0.1)
    cbar1 = plt.colorbar(im1, cax = cax1, orientation = 'vertical')
    cbar1.set_ticks([ 0.0, np.pi/4.0, np.pi/2.0])
    cbar1.set_ticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"])
    cbar1.set_label(r"$\langle\theta_{\mathbf{v}_i, \mathbf{v}_j}\rangle$")
    ax11  = div1.append_axes('left', size = '5%', pad = 0.8, sharey = ax1)
    ax12  = div1.append_axes('bottom', size = '8%', pad = 0.8, sharex = ax1)
    for i in range(1, dof + 1):
        if i == zeroindx:
            ax11.plot(0.5, i + 0.5, markerfacecolor = 'black', markeredgecolor = 'black', marker = 'o')
            ax12.plot(i + 0.5, 0.5, markerfacecolor = 'black', markeredgecolor = 'black', marker = 'o', label = r"$\lambda_i = 0$", c = "w")
        elif i < zeroindx: 
            ax11.plot(0.5, i + 0.5, markerfacecolor = 'blue', markeredgecolor = 'blue', marker = 'o', alpha = (dof - i)/dof)
            ax12.plot(i + 0.5, 0.5, markerfacecolor = 'blue', markeredgecolor = 'blue', marker = 'o', alpha = (dof - i)/dof, label = r"$\lambda_i > 0$", c = "w")
        elif i > zeroindx: 
            ax11.plot(0.5, i + 0.5, markerfacecolor = 'red', markeredgecolor = 'red', marker = 'o', alpha = (i)/dof)
            ax12.plot(i + 0.5, 0.5, markerfacecolor = 'red', markeredgecolor = 'red', marker = 'o', alpha = (i)/dof, label = r"$\lambda_i < 0$", c = "w")
    ax11.set_ylim(1, dof)
    ax12.set_xlim(1, dof)
    ax11.axis("off")
    ax12.axis("off")
    ax12.legend(loc = "lower right", bbox_to_anchor = (-0.01, -0.1), fancybox = True, framealpha = 1, shadow = True)
    
    
    ## Time averaged squared components of the vectors
    ax2  = fig.add_subplot(gs[0, 1])
    im  = ax2.imshow(np.flipud(zdata), cmap = cm.jet, extent = [1, dof, kmin, num_osc])
    ax2.set_xlabel(r"$j$")
    ax2.set_ylabel(r"$k$")
    ax2.set_title(r"Time-averaged Components$")
    div2  = make_axes_locatable(ax2)
    cax2  = div2.append_axes('right', size = '5%', pad = 0.1)
    cbar = plt.colorbar(im, cax = cax2, orientation = 'vertical')
    cbar.set_label(r"$\langle |\mathbf{v}_k^j |^2\rangle$")
    ax22  = div2.append_axes('bottom', size = '8%', pad = 0.8, sharex = ax2)
    for i in range(1, dof + 1):
        if i == zeroindx:
            ax22.plot(i + 0.5, 0.5, markerfacecolor = 'black', markeredgecolor = 'black', marker = 'o', label = r"$\lambda_i = 0$", c = "w")
        elif i < zeroindx: 
            ax22.plot(i + 0.5, 0.5, markerfacecolor = 'blue', markeredgecolor = 'blue', marker = 'o', alpha = (dof - i)/dof, label = r"$\lambda_i > 0$", c = "w")
        elif i > zeroindx: 
            ax22.plot(i + 0.5, 0.5, markerfacecolor = 'red', markeredgecolor = 'red', marker = 'o', alpha = (i)/dof, label = r"$\lambda_i < 0$", c = "w")
    ax22.set_xlim(1, dof)
    ax22.axis("off")
    
    
    ## Distrubution between stable and unstable manifolds
    ax3 = fig.add_subplot(gs[1, 0:])
    hist, bins  = np.histogram(theta1, range = (0.0, np.pi / 2.0), bins = 900, density = True)
    bin_centers = (bins[1:] + bins[:-1]) * 0.5
    plt.plot(bin_centers, hist)
    if zeroindx > 0:
        hist, bins  = np.histogram(theta2, range = (0.0, np.pi / 2.0), bins = 900, density = True)
        bin_centers = (bins[1:] + bins[:-1]) * 0.5
        ax3.plot(bin_centers, hist)
    ax3.set_xlim(0.0, np.pi/2.0)
    ax3.set_xlabel(r"$\theta$")
    ax3.set_xticks([0.0, np.pi/4.0, np.pi/2.0],[r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"])
    ax3.set_ylabel(r"PDF")
    ax3.set_yscale("log")
    ax3.legend([r"$\theta_{\mathbf{E}^+ \oplus \mathbf{E}^0, \mathbf{E}^-}$", r"$\theta_{\mathbf{E}^+, \mathbf{E}^0 \oplus \mathbf{E}^-}$"], fancybox = True, framealpha = 1, shadow = True)
    ax3.set_title(r"Distribution of Angles Between Tangent Subspaces")
    
    plt.suptitle(r"$N = {}, k_0 = {}, \alpha = {:0.3f}, \beta = {:0.3f}$".format(N, k0, alpha, beta))
    
    plt.tight_layout(rect = (0, 0, 1, 0.96))
    plt.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/CLVs" + "/CLV_Stats_N[{}]_ALPHA[{:0.3f}].png".format(N, alpha), format = "png")  
    plt.close

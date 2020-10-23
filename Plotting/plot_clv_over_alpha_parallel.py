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

def compute_angles_subspaces(A1, B1, num_clv_steps):
    
    angles1 = np.zeros((num_clv_steps))
    
    for t in range(num_clv_steps):
        angles1[t] = subspace_angles(A1[t, :, :], B1[t, :, :])[0]
        
    return angles1


def find_min_max(N, k0, alpha, beta, u0, iters, m_end, m_iter, trans, min_v_z, max_v_z):
    
    for a in alpha:

        ## ------- Read in Data
        ## Create filename from data
        filename = input_dir + "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/CLVData_ITERS[{},{},{}]".format(N, k0, a, beta, u0, iters, m_end, m_iter)

        ## Check if file exists and open
        if os.path.exists(filename + "_TRANS[{}].h5".format(trans)):
            HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans), 'r')
        elif os.path.exists(filename + "_TRANS[{}].h5".format(trans * 10)):
            HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans * 10), 'r')
        elif os.path.exists(filename + "_TRANS[{}].h5".format(trans / 10)):
            HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans / 10), 'r')
        elif os.path.exists(filename + "_TRANS[{}].h5".format(trans * 100)):
            HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans * 100), 'r')
        else: 
            print("File doesn't exist, check parameters!")
            sys.exit()

        ## --------- Read in datasets
        CLVs   = HDFfileData['CLVs']
        amps   = HDFfileData['Amps'][:]
        
        ## --------- System Parameters
        num_clv_steps = CLVs.shape[0]
        num_osc       = amps.shape[0];
        dof           = num_osc - (k0 + 1)

        ## --------- Reshape the CLV and Angles data
        clv_dims = CLVs.attrs['CLV_Dims']
        CLV      = np.reshape(CLVs, (CLVs.shape[0], dof, dof))
    
        ## --------- Compute time averaged squared vector components
        zdata  = compute_zdata(CLV, num_clv_steps, dof)
        
        ## --------- Find min and max values 
        min_z = np.amin(zdata)
        max_z = np.amax(zdata)

        if min_z < min_v_z:
            min_v_z = min_z
        if max_z > max_v_z:
            max_v_z = max_z

    return min_v_z, max_v_z


def loop_over_data(N, k0, a, beta, u0, iters, m_end, m_iter, trans, min_v_z, max_v_z):

    ## ------- Read in Data
    ## Create filename from data
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    filename = input_dir + "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/CLVData_ITERS[{},{},{}]".format(N, k0, a, beta, u0, iters, m_end, m_iter)

    ## Check if file exists and open
    if os.path.exists(filename + "_TRANS[{}].h5".format(trans)):
        HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans), 'r')
    elif os.path.exists(filename + "_TRANS[{}].h5".format(trans * 10)):
        HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans * 10), 'r')
    elif os.path.exists(filename + "_TRANS[{}].h5".format(trans / 10)):
        HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans / 10), 'r')
    elif os.path.exists(filename + "_TRANS[{}].h5".format(trans / 100)):
        HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans / 100), 'r')
    elif os.path.exists(filename + "_TRANS[{}].h5".format(trans * 100)):
        HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans * 100), 'r')
    else: 
        print("File doesn't exist, check parameters!")
        sys.exit()


    ## --------- Read in datasets
    # phases = HDFfileData['Phases'][:, :]
    time   = HDFfileData['Time'][:]
    amps   = HDFfileData['Amps'][:]
    lce    = HDFfileData['LCE'][:, :]
    CLVs   = HDFfileData['CLVs']
    angles = HDFfileData['Angles']

    ## --------- System Parameters
    num_tsteps    = len(time);
    num_clv_steps = CLVs.shape[0]
    num_osc       = amps.shape[0];
    kmin          = k0 + 1;
    kmax          = num_osc - 1;
    dof           = num_osc - kmin

    ## --------- Reshape the CLV and Angles data
    clv_dims = CLVs.attrs['CLV_Dims']
    CLV      = np.reshape(CLVs, (CLVs.shape[0], dof, dof))
    ang_dims = angles.attrs['Angle_Dims']
    angle    = np.reshape(angles, (angles.shape[0], dof, dof))


    ## --------- Compute Data
    ## Time averaged squared vector components
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
    else:
        theta2 = 0


    # ## --------- Plot all CLV Data
    # plot_data(angle, zdata, theta1, theta2, zeroindx, kmin, num_osc, dof, min_v_z, max_v_z, N, k0, a, beta)

    ## -------- Plot Time Averaged CLV plot
    plot_time_averaged_clv(zdata, a, N, zeroindx, dof, kmin, num_osc)


def plot_time_averaged_clv(zdata, a, N, zeroindx, dof, kmin, num_osc):
    
    fig = plt.figure(figsize = (24, 24))
    gs  = GridSpec(1, 1)
    
    cmap_new = cm.jet
    
    ## Time averaged squared components of the vectors
    ax2  = fig.add_subplot(gs[0, 0])
    im  = ax2.imshow(np.flipud(zdata), cmap = cm.jet, extent = [1, dof, kmin, num_osc]) # , vmin = 0.0, vmax = max_v_z
    ax2.set_xlabel(r"$j$")
    ax2.set_ylabel(r"$k$")
    ax2.set_title(r"Time-averaged Components$")
    div2  = make_axes_locatable(ax2)
    cax2  = div2.append_axes('right', size = '5%', pad = 0.1)
    cbar = plt.colorbar(im, cax = cax2, orientation = 'vertical')
    cbar.set_label(r"$\langle |\mathbf{v}_k^j |^2\rangle$")
    ax22  = div2.append_axes('bottom', size = '8%', pad = 0.8, sharex = ax2)
    for i in range(0, dof - 1):
        if i == zeroindx:
            ax22.plot(i + 1.5, 0.5, markerfacecolor = 'black', markeredgecolor = 'black', marker = 'o', label = r"$\lambda_i = 0$", c = "w")
        elif i < zeroindx: 
            ax22.plot(i + 1.5, 0.5, markerfacecolor = 'blue', markeredgecolor = 'blue', marker = 'o', alpha = (dof - i)/dof, label = r"$\lambda_i > 0$", c = "w")
        elif i > zeroindx: 
            ax22.plot(i + 1.5, 0.5, markerfacecolor = 'red', markeredgecolor = 'red', marker = 'o', alpha = (i)/dof, label = r"$\lambda_i < 0$", c = "w")
    ax22.set_xlim(1, dof)
    ax22.axis("off")   
    ax22.legend(loc = "lower right", bbox_to_anchor = (-0.01, -0.1), fancybox = True, framealpha = 1, shadow = True) 
    ax2.set_title(r"$\alpha = {:0.3f}$".format(a))

    plt.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/CLVs" + "/TimeAverage_CLV_N[{}]_ALPHA[{:0.3f}].png".format(N, a), format = "png")  
    plt.close
    
    return print("Plotted alpha = {:0.3f}".format(a))


def plot_data(angles, zdata, theta1, theta2, zeroindx, kmin, num_osc, dof, min_v_z, max_v_z, N, k0, alpha, beta):
    fig = plt.figure(figsize = (24, 24))
    gs  = GridSpec(2, 2)
    
    cmap_new = cm.jet
    
    ## Angles between vectors averaged over time
    ax1  = fig.add_subplot(gs[0, 0])
    data = np.mean(angles, axis = 0)
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
    for i in range(0, dof - 1):
        if i == zeroindx:
            ax11.plot(0.5, i + 1.5, markerfacecolor = 'black', markeredgecolor = 'black', marker = 'o')
            ax12.plot(i + 1.5, 0.5, markerfacecolor = 'black', markeredgecolor = 'black', marker = 'o', label = r"$\lambda_i = 0$", c = "w")
        elif i < zeroindx: 
            ax11.plot(0.5, i + 1.5, markerfacecolor = 'blue', markeredgecolor = 'blue', marker = 'o', alpha = (dof - i)/dof)
            ax12.plot(i + 1.5, 0.5, markerfacecolor = 'blue', markeredgecolor = 'blue', marker = 'o', alpha = (dof - i)/dof, label = r"$\lambda_i > 0$", c = "w")
        elif i > zeroindx: 
            ax11.plot(0.5, i + 1.5, markerfacecolor = 'red', markeredgecolor = 'red', marker = 'o', alpha = (i)/dof)
            ax12.plot(i + 1.5, 0.5, markerfacecolor = 'red', markeredgecolor = 'red', marker = 'o', alpha = (i)/dof, label = r"$\lambda_i < 0$", c = "w")
    ax11.set_ylim(1, dof)
    ax12.set_xlim(1, dof)
    ax11.axis("off")
    ax12.axis("off")
    ax12.legend(loc = "lower right", bbox_to_anchor = (-0.01, -0.1), fancybox = True, framealpha = 1, shadow = True)
    
    
    ## Time averaged squared components of the vectors
    ax2  = fig.add_subplot(gs[0, 1])
    im  = ax2.imshow(np.flipud(zdata), cmap = cm.jet, extent = [1, dof, kmin, num_osc]) # , vmin = 0.0, vmax = max_v_z
    ax2.set_xlabel(r"$j$")
    ax2.set_ylabel(r"$k$")
    ax2.set_title(r"Time-averaged Components$")
    div2  = make_axes_locatable(ax2)
    cax2  = div2.append_axes('right', size = '5%', pad = 0.1)
    cbar = plt.colorbar(im, cax = cax2, orientation = 'vertical')
    cbar.set_label(r"$\langle |\mathbf{v}_k^j |^2\rangle$")
    ax22  = div2.append_axes('bottom', size = '8%', pad = 0.8, sharex = ax2)
    for i in range(0, dof - 1):
        if i == zeroindx:
            ax22.plot(i + 1.5, 0.5, markerfacecolor = 'black', markeredgecolor = 'black', marker = 'o', label = r"$\lambda_i = 0$", c = "w")
        elif i < zeroindx: 
            ax22.plot(i + 1.5, 0.5, markerfacecolor = 'blue', markeredgecolor = 'blue', marker = 'o', alpha = (dof - i)/dof, label = r"$\lambda_i > 0$", c = "w")
        elif i > zeroindx: 
            ax22.plot(i + 1.5, 0.5, markerfacecolor = 'red', markeredgecolor = 'red', marker = 'o', alpha = (i)/dof, label = r"$\lambda_i < 0$", c = "w")
    ax22.set_xlim(1, dof)
    ax22.axis("off")
    
    
    ## Distrubution between stable and unstable manifolds
    ax3 = fig.add_subplot(gs[1, 0:])
    hist, bins  = np.histogram(theta1, range = (0.0, np.pi / 2.0), bins = 900, density = True);
    bin_centers = (bins[1:] + bins[:-1]) * 0.5
    plt.plot(bin_centers, hist)
    if zeroindx > 0:
        hist, bins  = np.histogram(theta2, range = (0.0, np.pi / 2.0), bins = 900, density = True);
        bin_centers = (bins[1:] + bins[:-1]) * 0.5
        ax3.plot(bin_centers, hist)
    ax3.set_xlim(0.0, np.pi/2.0)
    ax3.set_xlabel(r"$\theta$")
    ax3.set_xticks([0.0, np.pi/4.0, np.pi/2.0],[r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"]);
    ax3.set_ylabel(r"PDF")
    ax3.set_yscale("log")
    ax3.legend([r"$\theta_{\mathbf{E}^+ \oplus \mathbf{E}^0, \mathbf{E}^-}$", r"$\theta_{\mathbf{E}^+, \mathbf{E}^0 \oplus \mathbf{E}^-}$"], fancybox = True, framealpha = 1, shadow = True)
    ax3.set_title(r"Distribution of Angles Between Tangent Subspaces")
    
    plt.suptitle(r"$N = {}, k_0 = {}, \alpha = {:0.3f}, \beta = {:0.3f}$".format(N, k0, alpha, beta))
    
    plt.tight_layout(rect = (0, 0, 1, 0.96))
    plt.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/CLVs" + "/CLV_Stats_N[{}]_ALPHA[{:0.3f}].png".format(N, alpha), format = "png")  
    plt.close

    return print("Plotted alpha = {:0.3f}".format(alpha))







if __name__ == '__main__':
    #########################
    ##  Parameter Space
    #########################
    k0     = 1
    # alpha  = np.arange(0.0, 3.5, 0.05)
    alpha  = np.array([0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.45])
    beta   = 0.0
    iters  = 400000
    trans  = 1000
    N      = 256
    u0     = "RANDOM"
    m_end  = 20000
    m_iter = 20


    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    
    min_v_z = 100000
    max_v_z = 0
    # #######################################
    # ##  First pass: Find min and max vals
    # #######################################
    # 

    # print("First Pass over data - find min and max values of CLV data!")
    # min_v_z, max_v_z = find_min_max()


    #######################################
    ##  Second pass: Plot Data
    #######################################
    print("Second Pass over data - plotting data!")
    
    ## Create Process list  
    procLim  = 9
    
    
    #########################
    ## Plot all CLV Data
    ## Create iterable group of processes
    groups_args =  [(mprocs.Process(target = loop_over_data, args = (N, k0, a, beta, u0, iters, m_end, m_iter, trans, min_v_z, max_v_z)) for a in alpha)] * procLim
    
      
    
    
    ## Loop of grouped iterable
    for procs in zip_longest(*groups_args): 
        processes = []
        for p in filter(None, procs):
            processes.append(p)
            p.start()

        for process in processes:
            process.join()

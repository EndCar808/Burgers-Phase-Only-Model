#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros

######################
##  Library Imports
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file

import matplotlib.pyplot as plt
# plt.style.use('seaborn-talk')
mpl.rcParams['figure.figsize'] = [16, 9]
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size']   = 24
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
mpl.rcParams['lines.linewidth'] = 1.25
mpl.rcParams['lines.markersize'] = 6
from scipy.io import FortranFile
kr='float64'
ki='int64'
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
from numba import jit, njit

# @njit
def compute_second_moment(a_k, k0, N):

    dx           = np.pi / N 
    kx           = np.arange(N + 1)
    kx[:k0 + 1]  = 0.0
    a_k_sqr      = (a_k ** 2) * (1.0 / float(N**2))
    sec_mmnt = np.zeros((N))

    for i in range(1, N + 1):
        sec_mmnt[i - 1] = np.sum(a_k_sqr * (1.0 - np.cos( i * dx * kx)))

    return sec_mmnt

def compute_modes_real_space(amps, phases, N):
    print("\n...Creating Real Space Soln...\n")

    # Create full set of amps and phases
    amps_full   = np.append(amps[:], np.flipud(amps[1:-1]))
    phases_full = np.concatenate((phases[:, :], -np.fliplr(phases[:, 1:-1])), axis = 1)

    # Construct modes and realspace soln
    u_z = amps_full * np.exp(np.complex(0.0, 1.0) * phases_full)
    u   = np.real(np.fft.ifft(u_z, axis = 1))

    return u, u_z

@njit
def compute_rms(u):
    
    # Compute normalized realspace soln
    u_urms = np.zeros((u.shape[0], u.shape[1]))
    u_rms  = np.sqrt(np.mean(u[10, :]**2))
    
    for i in range(u.shape[0]):    
        u_urms[i, :] = u[i, :] / u_rms
    
    return u_rms, u_urms



def compute_grad(u_z, kmin, kmax):
    print("\nCreating Gradient\n")
    k            = np.concatenate((np.zeros((kmin)), np.arange(kmin, kmax + 1), -np.flip(np.arange(kmin, kmax)), np.zeros((kmin - 1))))
    grad_u_z     = np.complex(0.0, 1.0) * k * u_z
    du_x         = np.real(np.fft.ifft(grad_u_z, axis = 1))
    du_x_rms_tmp = np.sqrt(np.mean(du_x[10, :] ** 2))
    du_x_rms     = du_x / du_x_rms_tmp

    return du_x , du_x_rms


@jit(nopython = True)
def compute_velinc(u, rlen):

    # rList  = [int(r) for r in np.arange(1, kmax + 1, kmax/rlen)]
    rList  = [1, kmax]
    du_r   = np.zeros((u.shape[0], u.shape[1], rlen))

    for r_indx, r in enumerate(rList):
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                    du_r[i, j, r_indx] = u[i, np.mod(j + r, u.shape[1])] - u[i, j]

    return du_r, rList


def load_parameters(base_dir='../',
                    name='1'
                    ):
    par=np.loadtxt(base_dir+'parameters/Parameters_{0}.dat'.format(name))
    N=int(par[0])
    k0=int(par[1])
    alpha=float(par[2])
    beta=float(par[3])
    n_max=int(par[4])
    n_bins=int(par[5])
    x_lim=float(par[6])
    dur_norm=float(par[7])
    pars={
    'N':N,
    'k0':k0,
    'alpha':alpha,
    'beta':beta,
    'n_max':n_max,
    'n_bins':n_bins,
    'x_lim':x_lim,
    'dur_norm':dur_norm
    }    
    return pars


if __name__ == '__main__':

    #########################
    ##  Get Input Parameters
    #########################
    if (len(sys.argv) != 8):
        print("No Input Provided, Error.\nProvide: \nk0\nAlpha\nBeta\nIterations\nTransient Iterations\nN\nu0\n")
        sys.exit()
    else: 
        k0    = int(sys.argv[1])
        alpha = float(sys.argv[2])
        beta  = float(sys.argv[3])
        iters = int(sys.argv[4])
        trans = int(sys.argv[5])
        N     = int(sys.argv[6])
        u0    = str(sys.argv[7])

    filename = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(N, k0, alpha, beta, u0, iters, trans)

    num_osc = int(N / 2 + 1)
    kmax    = num_osc - 1
    kmin    = k0 + 1
    num_obs = N * iters
    ######################
    ##  Input & Output Dir
    ######################
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/Stats"


    ######################
    ##  Open Input File
    ######################
    HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')

    # print input file name to screen
    print("\n\nData File: %s.h5\n" % filename)
    print(list(HDFfileData.keys()))


    ######################
    ##  Read in Datasets
    ######################   
    phases = HDFfileData["Phases"][:, :]
    amps   = HDFfileData["Amps"][:]
    vel_inc_bincounts = np.zeros((2, 8000))
    vel_inc_binedges  = np.zeros((2, 8001))
    for i in range(2):
        vel_inc_bincounts[i, :] = HDFfileData["VelInc[{}]_BinCounts".format(i)][:]
        vel_inc_binedges[i, :] = HDFfileData["VelInc[{}]_BinEdges".format(i)][:]
    grad_bincounts = HDFfileData["VelGrad_BinCounts"][:]
    grad_binedges  = HDFfileData["VelGrad_BinEdges"][:]
    
    u, u_z        = compute_modes_real_space(amps, phases, N)
    u_rms, u_urms = compute_rms(u)
    
    du_r, rlist   = compute_velinc(u_urms, 2)

    rms = np.sqrt(np.mean(amps ** 2))




    # subname = "RealSpace"
    # f= open(a_data_dir+'realspace/'+subname+'_'+name+'.dat', 'r')
    # RS = np.fromfile(f, dtype = kr, count = -1).reshape((iters, N + 1), order = "F")
    # # RS=f.read_reals(kr)#.reshape((iters, N), order = "F")
    # # f.close
    # f=FortranFile(a_data_dir+'realspace/'+subname+'_'+name+'.dat', 'r')
    # RS=f.read_reals(kr)
    # f.close
    # print(RS.shape)
    # print(RS[:5])
    # # print(RS[:5, :5])
    # # print(RS[:5, -5:-1])


    # rs = HDFfileData["RealSpace"][:, :]
    # print(rs.shape)
    # print(rs[:5, :5])
    # print(rs[:5, -5:-1])
    # err = np.zeros(u.shape[0])
    # for i in range(u.shape[0]):
    #     err[i] = np.linalg.norm(u[i, :] - RS[i, :], ord = 1)

    # plt.plot(RS, '-.')
    # plt.savefig(output_dir + "/RS_ERROR_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{}]_TRANS[{}].png".format(N, k0, alpha, beta, u0, iters, trans), format='png', dpi = 400)  
    # plt.close()


    a_data_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Agustin_Code/phase-oscillator-master_Copy_ECEdits_2/phase-oscillator-master/source/"
    name = "N10_k0_1"
    subname = "Grad"
    f=FortranFile(a_data_dir + 'histograms/bins_'+subname+'_'+name+'.dat', 'r')
    bins=f.read_reals(kr)
    f.close

    f=FortranFile(a_data_dir + 'histograms/Hist_'+subname+'_'+name+'.dat', 'r')
    H=f.read_reals(ki)
    f.close

    pars = load_parameters(a_data_dir, name)


    dx=(bins[1]-bins[0])
    args =np.argwhere(H != 0)
    bins=bins[args]
    H=H[args]
    H=H/np.sum(H*dx)
    H/=float(pars['dur_norm'])

    # d = 1
    # bin_centers = (vel_inc_binedges[d, 1:] + vel_inc_binedges[d, :-1]) * 0.5
    # bin_width   = vel_inc_binedges[d, 1] - vel_inc_binedges[d, 0]
    # args = np.argwhere(vel_inc_bincounts[d, :] != 0)
    bin_centers = (grad_binedges[1:] + grad_binedges[:-1]) * 0.5
    bin_width   = grad_binedges[1] - grad_binedges[0]
    bin_centers = bin_centers[args]
    # counts = vel_inc_bincounts[d, args]
    counts = grad_bincounts[args]
    # plt.plot(bin_centers, counts / np.sum(vel_inc_bincounts[d, :] * bin_width)/ float(pars['dur_norm']))
    plt.plot(bin_centers, counts / np.sum(grad_bincounts[:] * bin_width)/ float(pars['dur_norm']))
    plt.plot(bins, H)
    plt.legend([r"A", r"Mine"])
    plt.yscale('log')
    plt.grid(True)


    plt.savefig(output_dir + "/PDF_Vel_Incrments_COMPARE_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{}]_TRANS[{}].png".format(N, k0, alpha, beta, u0, iters, trans), format='png', dpi = 400)  
    plt.close()


    ######################
    ##  Vel Inc
    ######################   
    plt.figure()
    for i in range(2):
        bin_centers = (vel_inc_binedges[-i, 1:] + vel_inc_binedges[-i, :-1]) * 0.5
        bin_width   = vel_inc_binedges[-i, 1] - vel_inc_binedges[-i, 0]
        plt.plot(bin_centers, vel_inc_bincounts[-i] / (num_obs * bin_width) / float(pars['dur_norm']))
        # plt.plot(bin_centers, vel_inc_bincounts[i] / np.sum(vel_inc_bincounts[i] * bin_width))
        # plt.plot(vel_inc_bincounts[i] / (num_obs * bin_width))
        # print(vel_inc_bincounts[i])
    bin_centers = (grad_binedges[1:] + grad_binedges[:-1]) * 0.5
    bin_width   = grad_binedges[1] - grad_binedges[0]
    plt.plot(bin_centers, grad_bincounts / (num_obs * bin_width) / float(pars['dur_norm']))
    # plt.plot(bin_centers, grad_bincounts / np.sum(grad_bincounts * bin_width))
    # plt.plot(grad_bincounts / (num_obs * bin_width))
    plt.legend([r"Largest", r"Smallest", r"Gradient"])
    plt.yscale('log')
    # plt.xlim(-5, 5)
    plt.grid(True)


    plt.savefig(output_dir + "/PDF_Vel_Incrments_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{}]_TRANS[{}].png".format(N, k0, alpha, beta, u0, iters, trans), format='png', dpi = 400)  
    plt.close()

    
    ######################
    ##  Triad Centroid
    #####################   
    # Triad_Centroid = HDFfileData["TriadCentroid"]
    # Triad_Cent_R   = HDFfileData["TriadCentroid_R"]

    # # Reshape triads
    # tdims    = Triad_Centroid.attrs['Triad_Dims']
    # k_range  = tdims[0, 0] # kmax - kmin + 1
    # k1_range = tdims[0, 1] # int(k_range / 2)

    # triad_cent   = np.reshape(Triad_Centroid[:], (k_range, k1_range))
    # Triad_Cent_R = np.reshape(Triad_Cent_R[:], (k_range, k1_range))

    # fig = plt.figure(figsize = (16, 9), tight_layout = True)
    # gs  = GridSpec(1, 1)

    # myjet   = cm.jet(np.arange(255))
    # norm    = mpl.colors.Normalize(vmin = 0.0, vmax = 1.0)
    # my_mjet = mpl.colors.LinearSegmentedColormap.from_list('my_map', myjet, N = kmax) # set N to inertial range
    # my_mjet.set_under('1.0')
    # m       = cm.ScalarMappable(norm = norm, cmap = my_mjet)     

    # fig = plt.figure(figsize = (16, 9), tight_layout = True)
    # gs  = GridSpec(1, 1)

    # ax4  = fig.add_subplot(gs[0, 0])
    # im   = ax4.imshow(np.flipud(np.transpose(Triad_Cent_R)), cmap = my_mjet, norm = norm)
    # kMax = kmax - kmin # Adjusted indices in triads matrix
    # kMin = kmin - kmin # Adjusted indices in triads matrix
    # ax4.set_xticks([kmin, int((kMax - kMin)/5), int(2 * (kMax - kMin)/5), int(3* (kMax - kMin)/5), int(4 * (kMax - kMin)/5), kMax])
    # ax4.set_xticklabels([kmin, int((kmax - kmin)/5), int(2 * (kmax - kmin)/5), int(3* (kmax - kmin)/5), int(4 * (kmax - kmin)/5), kmax])
    # ax4.set_yticks([kMin, int((kMax / 2 - kMin)/4), int(2 * (kMax / 2 - kMin)/4), int(3* (kMax / 2 - kMin)/4),  int((kmax)/ 2 - kmin)])
    # ax4.set_yticklabels(np.flip([kmin + kmin, int((kmax / 2 - kmin)/4) + kmin, int(2 * (kmax / 2 - kmin)/4) + kmin, int(3* (kmax / 2 - kmin)/4) + kmin,  int(kmax / 2)]))
    # ax4.set_xlabel(r'$k$', labelpad = 0)
    # ax4.set_ylabel(r'$p$',  rotation = 0, labelpad = 10)
    # ax4.set_xlim(left = kmin - 0.5)
    # ax4.set_ylim(bottom = int((kmax)/ 2 - kmin) + 0.5)
    # div4  = make_axes_locatable(ax4)
    # cax4  = div4.append_axes('right', size = '5%', pad = 0.1)
    # cbar4 = plt.colorbar(im, cax = cax4, orientation='vertical')
    # cbar4.set_ticks([ 0.0, 0.5, 1])
    # cbar4.set_ticklabels([r"$0$", r"$0.5$", r"$1$"])
    # cbar4.set_label(r"$\mathcal{R}_{k, p}$")

    # plt.savefig(output_dir + "/Triad_Centroid_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{}]_TRANS[{}].png".format(N, k0, alpha, beta, u0, iters, trans), format='png', dpi = 400)  
    # plt.close()



   


    ######################
    ##  Str Funcs Plots
    ######################   
    # str_func     = HDFfileData["StructureFuncs"][:, :]

    # print(str_func.shape)
    # r = np.arange(0, N/2)
 
    # second_moment = compute_second_moment(amps, k0, int(N/2))

    # L = (N / 2)


    # urms = np.load("urms.npy")
    # increment2 = np.load("increment_moment_2.npy")

    # control_moment = np.load("control_second_moment.npy")

    # # print(urms)
    # # print(u_rms)

    # # # print(urms/u_urms)
    # plt.figure()
    # plt.plot(r/L, increment2 / urms**2)
    # plt.plot(r/L, np.absolute(str_func[0, :]) / (u_rms**( 2)))
    # plt.plot(r/L, second_moment[:] / u_rms**2, 'k--')
    # # plt.plot(r/L, control_moment / urms**2, )
    # # # print(urms)
    # # print(second_moment / increment2)
    # # print(second_moment[:5])
    # # print(control_moment[:5])
    # plt.yscale('Log')
    # plt.xscale('Log')  
    # plt.legend([r"Structure Func p = 2", r"My 2nd Moment"])

    # plt.savefig(output_dir + "/TEST_Structure_Funcs_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{}]_TRANS[{}].png".format(N, k0, alpha, beta, u0, iters, trans), format='png', dpi = 400)  
    # plt.close()


    # plt.figure()
    # for i in range(str_func.shape[0]):
    #     plt.plot(r/L, np.absolute(str_func[i, :]) / (u_rms**(i + 2)))
    # plt.plot(r/L, second_moment[:] / u_rms**2, 'k--')
   
    # plt.legend(np.append([r"$p = {}$".format(p) for p in range(2, 6)], r"Second Moment"))
    # plt.yscale('Log')
    # plt.xscale('Log')
    # plt.xlabel(r"$r / L$")
    # plt.ylabel(r"$|S^p(r)|/u_{rms}^p$")

    
    # plt.savefig(output_dir + "/Structure_Funcs_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{}]_TRANS[{}].png".format(N, k0, alpha, beta, u0, iters, trans), format='png', dpi = 400)  
    # plt.close()

    
    ######################
    ##  Space Time Plots
    ######################   
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    # fig.suptitle(r'$N = {} \quad \alpha = {} \quad \beta = {} \quad k_0 = {}$'.format(N, alpha, beta, k0))

    # ## REAL SPACE
    # im1 = ax1.imshow(np.flipud(u_urms), cmap = "bwr", extent = [0, N, 0, u_urms.shape[0]])
    # ax1.set_aspect('auto')
    # ax1.set_title(r"Real Space")
    # ax1.set_xlabel(r"$x$")
    # ax1.set_ylabel(r"$t$")
    # ax1.set_xticks([0.0, np.ceil(N / 4), np.ceil(2 * N / 4), np.ceil(3 * N / 4), N])
    # ax1.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2 \pi$"])
    # div1  = make_axes_locatable(ax1)
    # cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
    # cb1   = plt.colorbar(im1, cax = cbax1)
    # cb1.set_label(r"$u(x, t) / u^{rms}(x, t)$")

    # ## PHASES
    # im2  = ax2.imshow(np.flipud(np.mod(phases[:, kmin:], 2.0*np.pi)), cmap = "Blues", vmin = 0.0, vmax = 2.0 * np.pi, extent = [kmin, kmax, 0, u_urms.shape[0]])
    # ax2.set_aspect('auto')
    # ax2.set_title(r"Phases")
    # ax2.set_xlabel(r"$k$")
    # ax2.set_ylabel(r"$t$")
    # div2  = make_axes_locatable(ax2)
    # cbax2 = div2.append_axes("right", size = "10%", pad = 0.05)
    # cb2   = plt.colorbar(im2, cax = cbax2)
    # cb2.set_ticks([ 0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi])
    # cb2.set_ticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
    # cb2.set_label(r"$\phi_k(t)$")

    # plt.tight_layout(rect = (0, 0, 1, 0.96))
    # plt.savefig(output_dir + "/SPACETIME_N[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_k0[{}]_ITERS[{}].png".format(N, alpha, beta, k0, iters), format='png', dpi = 400)  
    # plt.close()






    u_grad, u_grad_rms = compute_grad(u_z, kmin, kmax)

    u_grad         = (u_grad * (np.pi / N/2)) / np.std((u_grad * (np.pi / N/2)).flatten())
    u_grad_rms     = u_grad_rms / np.std(u_grad_rms.flatten())

    plt.figure()
    hist, bins         = np.histogram(u_grad.flatten(), bins = 1000, density = False)
    bin_centers        = (bins[1:] + bins[:-1]) * 0.5
    bin_width          = bins[1] - bins[0]
    plt.plot(bin_centers, hist / (num_obs * bin_width))

    bin_centers = (grad_binedges[1:] + grad_binedges[:-1]) * 0.5
    bin_width   = grad_binedges[1] - grad_binedges[0]
    plt.plot(bin_centers, grad_bincounts / (num_obs * bin_width))
    plt.legend([r"Computed Grad", r"Solver Grad"])
    
    plt.yscale('log')
    # plt.xlim(-5, 5)
    plt.savefig(output_dir + "/Gradient_Test.png".format(N, alpha, beta, k0, iters), format='png', dpi = 400)  
    plt.close()


    plt.figure()
    small_scale = du_r[:, :, 0] / np.std(du_r[:, :, 0].flatten())
    hist, bins         = np.histogram(small_scale.flatten(), bins = 1000, density = False)
    bin_centers        = (bins[1:] + bins[:-1]) * 0.5
    bin_width          = bins[1] - bins[0]
    plt.plot(bin_centers, hist / (num_obs * bin_width))

    large_scale = du_r[:, :, 1] / np.std(du_r[:, :, 1].flatten())
    hist, bins         = np.histogram(large_scale.flatten(), bins = 1000, density = False)
    bin_centers        = (bins[1:] + bins[:-1]) * 0.5
    bin_width          = bins[1] - bins[0]
    plt.plot(bin_centers, hist / (num_obs * bin_width))
    
    plt.legend([r"$r = \pi / N$", r"$r = \pi$"])
    plt.xlabel(r"$\delta u_r / \sigma$")
    plt.ylabel(r"PDF")
    
    plt.yscale('log')
    # plt.xlim(-5, 5)
    plt.savefig(output_dir + "/Vel_Inc_Test.png".format(N, alpha, beta, k0, iters), format='png', dpi = 400)  
    plt.close()



    hist, bins         = np.histogram(small_scale.flatten(), bins = 1000, density = False)
    bin_centers        = (bins[1:] + bins[:-1]) * 0.5
    bin_width          = bins[1] - bins[0]
    plt.plot(bin_centers, hist / (num_obs * bin_width))

    hist, bins         = np.histogram(large_scale.flatten(), bins = 1000, density = False)
    bin_centers        = (bins[1:] + bins[:-1]) * 0.5
    bin_width          = bins[1] - bins[0]
    plt.plot(bin_centers, hist / (num_obs * bin_width))

    hist, bins         = np.histogram(u_grad.flatten(), bins = 1000, density = False)
    bin_centers        = (bins[1:] + bins[:-1]) * 0.5
    bin_width          = bins[1] - bins[0]
    plt.plot(bin_centers, hist / (num_obs * bin_width))

    hist, bins         = np.histogram(u_grad_rms.flatten(), bins = 1000, density = False)
    bin_centers        = (bins[1:] + bins[:-1]) * 0.5
    bin_width          = bins[1] - bins[0]
    plt.plot(bin_centers, hist / (num_obs * bin_width))

    plt.legend([r"Small", r"Large", r"Gradient", r"Gradient RMS"])
    plt.yscale('log')

    plt.savefig(output_dir + "/Vel_Inc+Grad_Test.png".format(N, alpha, beta, k0, iters), format='png', dpi = 400)  
    plt.close()


    # # normal_dist = np.random.normal(loc = 0, scale = 1, size = 100 * num_obs)
    # # hist, bins  = np.histogram(normal_dist, range = (-5, 5), bins = 1000, density = True)
    # # bin_centers = (bins[1:] + bins[:-1]) * 0.5
    # # plt.plot(bin_centers, hist, 'k--')
    
    hist, bins         = np.histogram(large_scale.flatten(), bins = 1000, density = True)
    bin_centers        = (bins[1:] + bins[:-1]) * 0.5
    bin_width          = bins[1] - bins[0]
    # plt.plot(bin_centers, hist / (num_obs * bin_width))
    plt.plot(bin_centers, hist)

    # large_scale_bins = bin_centers
    # large_scale_pdf  = hist / (num_obs * bin_width)
    # np.save("large_scale_bins.npy", large_scale_bins)
    # np.save("large_scale_pdf.npy", large_scale_pdf)

    hist, bins         = np.histogram(small_scale.flatten(), bins = 1000, density = True)
    bin_centers        = (bins[1:] + bins[:-1]) * 0.5
    bin_width          = bins[1] - bins[0]
    # plt.plot(bin_centers, hist / (num_obs * bin_width))
    plt.plot(bin_centers, hist)
   
    # small_scale_bins = bin_centers
    # small_scale_pdf  = hist / (num_obs * bin_width)
    # np.save("small_scale_bins.npy", small_scale_bins)
    # np.save("small_scale_pdf.npy", small_scale_pdf)

    hist, bins         = np.histogram(u_grad_rms.flatten(), bins = 1000, density = True)
    bin_centers        = (bins[1:] + bins[:-1]) * 0.5
    bin_width          = bins[1] - bins[0]
    # plt.plot(bin_centers, hist / (num_obs * bin_width))
    plt.plot(bin_centers, hist)




    plt.legend([r"Large-Scale", r"Small-Scale", r"Gradient"])
    plt.yscale('log')

    plt.savefig(output_dir + "/Vel_Inc+Grad.png".format(N, alpha, beta, k0, iters), format='png', dpi = 400)  
    plt.close()
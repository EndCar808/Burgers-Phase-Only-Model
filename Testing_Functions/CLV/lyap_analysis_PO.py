#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros

######################
##  Library Imports
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# plt.style.use('seaborn-talk')
mpl.rcParams['figure.figsize']    = [10, 8]
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['text.usetex']       = True
mpl.rcParams['font.family']       = 'serif'
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
np.set_printoptions(threshold=sys.maxsize)
from numba import njit

import pyfftw as fftw




###########################
##  Function Definitions
###########################
@njit
def initial_conditions(num_osc, alpha, beta, k0, dof, numLEs, u0):

    phi  = np.zeros((num_osc, ))
    amp  = np.zeros((num_osc, ))
    pert = np.zeros((dof, numLEs))

    for i in range(num_osc):
        if u0 == "RANDOM":       
            if i <= k0:
                phi[i] = 0.0
                amp[i] = 0.0
            else:
                phi[i] = 2.0 * np.pi * np.random.rand()
                amp[i] = 1 / (i ** alpha)
        elif u0 == "TEST":
            if i <= k0:
                amp[i] = 0.0
                phi[i] = 0.0
            else:
                phi[i] = np.pi / 4.0
                amp[i] = 1 / (i ** alpha)
    
    for i in range(numLEs):
        pert[i, i] = 1.0
        
    return phi, amp, pert




def compute_time(N, alpha, beta, k0, iters):

    M       = int(2 * N)
    num_osc = int(N / 2 + 1)

    ## Create the wavenumber array
    kx = np.arange(0, (2*num_osc - 1))

    ## Create input and ouput fft arrays
    real = fftw.zeros_aligned(M, dtype = 'float64')
    cplx = fftw.zeros_aligned((2*num_osc - 1), dtype = 'complex128')

    ## Set up the fft scheme
    fft_r2c = fftw.builders.rfft(real)
    fft_c2r = fftw.builders.irfft(cplx)

    ## Fill the arrays
    for i in range(num_osc):
        if i > k0:
            cplx[i] = 1 / (kx[i] ** alpha) * np.exp(1j * 0.0)
            
    ## Execute inverse transform
    fft_r = fft_c2r()

    ## Square in real space
    fft_r = (fft_r * M) ** 2

    ## Excute transform back to Fourier space
    ifft_c = fft_r2c(fft_r)

    ## Get RHS
    rhs = (-kx[k0 + 1:num_osc] / np.absolute(cplx[k0 + 1:num_osc])) * np.real(np.exp(-1j * np.angle(cplx[k0 + 1:num_osc])) * (ifft_c[k0 + 1:num_osc] / M))


    ## Find timestep
    step = 1 / np.nanmax(np.absolute(rhs))

    ## Get transient iterations 
    trans_ratio = np.nanmax(np.absolute(rhs)) / np.nanmin(np.absolute(rhs))
    trans_mag   = int(np.ceil(np.log10(trans_ratio)) + 1)
    trans_steps = 10 ** trans_mag
    
    ## Generate the time array
    start_time = trans_steps * step
    end_time   = iters * step
    time       = np.arange(start_time, end_time, step)
    
    return time, step, trans_steps



@njit
def nonlinearRHS(u_z, u_z_conv, pert, dof, kmin, kmax, M):

    ## Allocate memory
    jac = np.zeros((dof, dof))

    ## Compute the jacobian
    for k in range(kmin, kmax + 1):
        for k1 in range(kmin, kmax + 1):
            if k == k1:
                jac[k - kmin, k1 - kmin] = (-k / 2) * np.imag((u_z_conv[k] / M) / u_z[k])
                if k + k1 < num_osc:
                    jac[k - kmin, k1 - kmin] += k * np.imag(u_z[k1] * np.conjugate(u_z[k + k1]) / np.conjugate(u_z[k]))
            else:
                if k - k1 <= -kmin and k - k1 >= -kmax:
                    jac[k - kmin, k1 - kmin] = k * np.imag(u_z[k1] * np.conjugate(u_z[np.absolute(k - k1)])/ u_z[k])
                elif k - k1 >= kmin and k - k1 <= kmax:
                    jac[k - kmin, k1 - kmin] = k * np.imag(u_z[k1] * u_z[k - k1]/ u_z[k])
                if k + k1 < num_osc:
                    jac[k - kmin, k1 - kmin] += k * np.imag(u_z[k1] * np.conjugate(u_z[k + k1]) / np.conjugate(u_z[k]))

    ## Map the tangent space vectors forward using the jacobian
    pert = np.dot(jac, pert)

    return pert




def PO_RHS(phi, tang, amps, num_osc, k0, dof, numLEs):

    ## Parameters
    N         = 2 * (num_osc - 1)
    M         = 2 * N
    M_num_osc = int(M / 2 + 1)
    kmin      = k0 + 1
    kmax      = num_osc - 1  

    ## Allocate memory
    rhs  = np.zeros((num_osc, ))
    pert = np.zeros((dof, num_osc))

    ## Create input and ouput fft arrays
    global u
    global u_z
    global fftw_r2c
    global fftw_c2r

    ## Write modes to padded array
    for i in range(M_num_osc):
        if i > k0 and i < num_osc:
            u_z[i] = amps[i] * np.exp(np.complex(0.0, 1.0) * phi[i])
        else:
            u_z[i] = np.complex(0.0, 0.0)

    ## Execute Inverse transform
    u = fftw_c2r(u_z)

    ## Square in real space
    u = (u * M) ** 2

    ## Excute transform back to Fourier space
    u_z_conv = fftw_r2c(u)

    ## Compute RHS using the compute convolution
    for i in range(k0 + 1, num_osc):
        rhs[i] = (-i / np.absolute(u_z[i])) * np.real(np.exp(np.complex(0.0, -1.0) * np.angle(u_z[i])) * (u_z_conv[i] / M))

    ## Compute the rhs for the tangent dynamcis
    pert = nonlinearRHS(u_z, u_z_conv, tang, dof, kmin, kmax, M)

    return rhs, pert




# @njit
def RKsolver(phase_space, tang_space, amp, RHS, dt, num_osc, k0, dof, numLEs):

    ## Allocate memory
    rk1 = np.zeros((num_osc, ))
    rk2 = np.zeros((num_osc, ))
    rk3 = np.zeros((num_osc, ))
    rk4 = np.zeros((num_osc, ))
    rk1_pert = np.zeros((dof, numLEs))
    rk2_pert = np.zeros((dof, numLEs))
    rk3_pert = np.zeros((dof, numLEs))
    rk4_pert = np.zeros((dof, numLEs))


    ## Stages
    rk1, rk1_ext = RHS(phase_space, tang_space, amp, num_osc, k0, dof, numLEs)
    rk2, rk2_ext = RHS(phase_space + (0.5 * dt) * rk1, tang_space + (0.5 * dt) * rk1_ext, amp, num_osc, k0, dof, numLEs)
    rk3, rk3_ext = RHS(phase_space + (0.5 * dt) * rk2, tang_space + (0.5 * dt) * rk2_ext, amp, num_osc, k0, dof, numLEs)
    rk4, rk4_ext = RHS(phase_space + dt * rk3, tang_space + dt * rk3_ext, amp, num_osc, k0, dof, numLEs)
 
    ## Update steps
    # out     = phase_space + (dt / 6.0) * (rk1 + 2.0 * rk2 + 2.0 * rk3 + rk4)
    # out_ext = tang_space + (dt / 6.0) * (rk1_ext + 2.0 * rk2_ext + 2.0 * rk3_ext + rk4_ext) 
    out     = phase_space + (dt / 6.0) * rk1 + (dt / 3.0) * rk2 + (dt / 3.0) * rk3 + (dt / 6.0) * rk4
    out_ext = tang_space + (dt / 6.0) * rk1_ext + (dt / 3.0) * rk2_ext + (dt / 3.0) * rk3_ext + (dt / 6.0) * rk4_ext

    return out, out_ext



@njit 
def compute_LEs(rates, run_sum, LEs, t, t0, numLEs):

    for i in range(numLEs):
        run_sum[i] += np.log(rates[i])
        LEs[i]     = run_sum[i] / (t)

    return LEs, run_sum



def print_update(m, t, m_iters, m_end, LEs, numLEs):

    lce_sum = 0.0
    dim_k   = 0
    for i in range(numLEs):
        if lce_sum + LEs[i] >= 0:
            lce_sum += LEs[i]
            dim_k   += 1
        else:
            break
    print("Iter: {} | t = {:0.02f} / {} | Sum: {:5.6f} | Dim: {:5.6f}".format(m, t, m_iters * m_end, np.sum(LEs), dim_k + (lce_sum / np.absolute(LEs[dim_k]))))
    print("h = {}".format(LEs))



# @njit
def forward_dynamics(x, x_ext, amp, RHS, dt, m_iters, m_end, m_trans, m_rev_trans, dof, numLEs):

    ## Allocate memory
    LEs     = np.zeros(numLEs)
    rates   = np.zeros(numLEs)
    run_sum = np.zeros(numLEs)

    ## CLV arrays
    R_tmp     = np.zeros((m_end, dof, numLEs))
    Tang_vecs = np.zeros((m_end - m_rev_trans, dof, numLEs))

    ## Output arrays
    x_out     = np.zeros((m_end, num_osc))
    x_ext_out = np.zeros((m_end, dof, numLEs))

    ## Iteration counters
    m     = 0
    iters = 1

    ## Begin algorithm
    while m < m_trans + m_end:

        ## Integrate the system forward
        for i in range(m_iters):
        
            ## Call solver
            x, x_ext = RKsolver(x, x_ext, amp, RHS, dt, num_osc, k0, dof, numLEs)

            for i in range(num_osc):
                print("phi[{}]: {}".format(i, x[i]))

            for i in range(dof):
                for j in range(numLEs):
                    print("pert[{}]: {}".format(i * numLEs + j, x_ext[i, j]))

            ## Update iteration counter
            t     = iters * dt
            iters += 1


        ## Perform a QR decompostion of the tangent space
        Q, R = np.linalg.qr(x_ext)
        x_ext = Q

        ## Extrat growth rates - diagonals
        local_growth_rates = np.absolute(np.diag(R))


        
        ## If transient iterations have been reached - compute LEs
        if m >= m_trans:
            ## Record system state
            x_out[m - m_trans, :]     = x
            x_ext_out[m - m_trans, :] = x_ext

            ## Compute the current LEs
            LEs, run_sum = compute_LEs(local_growth_rates, run_sum, LEs, t, (m_trans) * dt, numLEs)

            ## Print update
            if np.mod(m, (m_end * 0.1)) == 0:
                print_update(m, t, m_iters, m_end, LEs, numLEs)

            ## Save the QR data for the CLVs
            R_tmp[m - m_trans, :, :]  = R
            if m < m_end - m_rev_trans:
                Tang_vecs[m - m_trans, :, :] = Q

        ## Update for next iteration
        m += 1
    print(iters)

    return x_out, x_ext_out, LEs, R_tmp, Tang_vecs


@njit
def backward_dynamics(R_tmp, GS_vecs, m_rev_iters, m_rev_trans, dof, numLEs):

    ## Allocate memory
    CLVs = np.zeros((m_rev_iters - m_rev_trans, dof, numLEs))

    ## Create random upper triangular matrix - coeffiecients of the CLVs
    # C = np.array([[1.0, 0.5, 1/3], [0, 0.5, 1/3], [0, 0, 1/3]])
    C = np.triu(np.random.rand(dof, numLEs))
    for i in range(dof):
        for j in range(numLEs):
            if j >= i:
                C[i, j] = 1 / (j + 1)
            else:
                C[i, j] = 0.0
    for i in range(numLEs):
        C[:, i] /= np.linalg.norm(C[:, i])


    rev_iters = 0
    for t in range(m_rev_iters - 1, -1, -1):

        ## Iterate coefficient matrix backward
        C = np.linalg.solve(R_tmp[t, :, :], C)

        ## Normalize
        for i in range(numLEs):
            C[:, i] /= np.linalg.norm(C[:, i])


        if t < m_rev_iters - m_rev_trans:

            ## Express the CLVs in the tangent space basis
            CLVs[t, :, :] = np.dot(GS_vecs[t, :, :], C)
            rev_iters += 1

    return CLVs


###########################
##  Main
###########################
if __name__ == '__main__':

    ## Lorenz setup parameters
    N       = int(sys.argv[1])
    k0      = int(sys.argv[2])
    alpha   = float(sys.argv[3])
    beta    = 0.0
    u0      = sys.argv[4]
    m_end   = int(sys.argv[5])
    m_iters = int(sys.argv[6])

    num_osc = int(N / 2 + 1)
    kmin    = k0 + 1
    kmax    = num_osc - 1
    dof     = int(num_osc - kmin)
    numLEs  = dof
    
    _, dt, trans_iters = compute_time(N, alpha, beta, k0, m_iters * m_end)
    m_trans     = int(trans_iters / m_iters)
    m_rev_trans = m_trans
    t0 = 0.0
    T  = dt * m_iters 

    ## Allocate global arrays for FFTW
    u   = fftw.zeros_aligned(2 * N, dtype = 'float64')
    u_z = fftw.zeros_aligned(N + 1, dtype = 'complex128')

    ## Set up the fft scheme
    fftw_r2c = fftw.builders.rfft(u)
    fftw_c2r = fftw.builders.irfft(u_z)

    phi  = np.zeros((num_osc, ))
    amp  = np.zeros((num_osc, ))
    pert = np.zeros((dof, numLEs))

    print(trans_iters)
    print(dt)

    # ## Get initial conditions
    phi, amp, pert = initial_conditions(num_osc, alpha, beta, k0, dof, numLEs, u0)
    for i in range(num_osc):
        print("phi[{}]: {:0.8f}\tamp[{}]: {:0.8f}\tu_z[{}]: {:4.8f}".format(i, phi[i], i, amp[i], i, amp[i] * np.exp(np.complex(0.0, 1.0) * phi[i])))
    print()

    # print(phi)
    # rk1, rk1_ext = PO_RHS(phi, pert, amp, num_osc, k0, dof, numLEs)

    # tmp1      = phi + 0.5 * dt * rk1
    # tmp1_pert = pert + (0.5 * dt) * rk1_ext
    # rk2, rk2_ext = PO_RHS(tmp1, tmp1_pert, amp, num_osc, k0, dof, numLEs)
    
    # tmp2      = phi + 0.5 * dt * rk2
    # tmp2_pert = pert + (0.5 * dt) * rk2_ext
    # rk3, rk3_ext = PO_RHS(tmp2, tmp2_pert, amp, num_osc, k0, dof, numLEs)
    
    # tmp3      = phi + dt * rk3
    # tmp3_pert = pert + (dt) * rk3_ext
    # rk4, rk4_ext = PO_RHS(tmp3, tmp3_pert, amp, num_osc, k0, dof, numLEs)
    # print("Rk1")
    # print(rk1)
    # print()

    # print("Rk2")
    # print(rk2)
    # print()
    # print("Rk3")
    # print(rk3)
    # print()
    # print("Rk4")
    # print(rk4)
    # print()


    # # print(phi)
    # # print()

    # # for i in range(dof):
    # #     for j in range(dof):
    # #         print("pert[{}]: {}\t".format(i * dof + j, pert[i, j]), end = "")
    # #     print()
    # # print()

    # phi, pert = RKsolver(phi, pert, amp, PO_RHS, dt, num_osc, k0, dof, numLEs)


    # print(phi)
    # print()

    phi, pert, LEs, R_tmp, GS = forward_dynamics(phi, pert, amp, PO_RHS, dt, m_iters, m_end, m_trans, m_rev_trans, dof, numLEs)

    print("Phi")
    print(phi[-1, :])
    print()

    print("Pert")
    for i in range(dof):
        for j in range(dof):
            print("pert[{}]: {}\t".format(i * dof + j, pert[-1, i, j]), end = "")
        print()
    print()
    
    print("LCE")
    for i in range(dof):
        print("LCE[{}]:{} ".format(i, LEs[i]))
    print()

    

    CLVs = backward_dynamics(R_tmp, GS, m_end, m_rev_trans, dof, numLEs)


    for i in range(dof):
        print("Row: {}".format(i))
        for j in range(numLEs):
            print("CLVs[{}]: {}".format(i * numLEs + j, CLVs[-1, i, j]))
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



###########################
##  Function Definitions
###########################
@njit
def initial_conditions(dim, sys):

    x     = np.zeros((dim))
    x_ext = np.identity(dim)
    if sys == "Lorenz63a":       
        x[0] = -1.01
        x[1] = 3.01
        x[2] = 2.01 
    elif sys == "Lorenz63b":  
        x[0] = 1.73000
        x[1] = 3.23000
        x[2] = 8.01000 
    elif sys == "Henon2D" or sys == "Lozi2D" or sys == "Henon2D_Alt":
        x[0] = 0.5
        x[1] = 0.5
    elif sys == "Henon3D":
        x[0] = 0.5
        x[1] = 0.5
        x[2] = 0.5

    return x, x_ext



@njit
def Lorenz_RHS(x, x_ext):
    
    ## Create RHS array
    f = np.zeros(x.shape)

    ## Fill RHS
    f[0] = sigma * (x[1] - x[0]) 
    f[1] = x[0] * (rho - x[2]) - x[1]
    f[2] = x[0] * x[1] - beta *  x[2]

    ## Create jacobian array
    Jac = np.array([[-sigma, sigma, 0], [rho - x[2], -1, -x[0]], [x[1], x[0], -beta]])

    ## Fill tangent space array
    f_ext = np.dot(Jac, x_ext)

    return f, f_ext


def PO_RHS(phi, pert, amps, num_osc, k0, dof, numLEs):

    ## Parameters
    N         = 2 * (num_osc - 1)
    M         = 2 * M
    M_num_osc = M / 2 + 1 

    ## Create input and ouput fft arrays
    u = fftw.zeros_aligned(M, dtype = 'float64')
    u_z = fftw.zeros_aligned(M_num_osc, dtype = 'complex128')

    ## Set up the fft scheme
    fft_r2c = fftw.builders.rfft(u)
    fft_c2r = fftw.builders.irfft(u_z)

    ## Write modes to padded array
    for i in range(M_num_osc):
        if i < num_osc:
            u_z[i] = amps[i] * np.exp(np.complex(0.0, 1.0) * phi[i])
        else:
            u_z[i] = np.comlex(0.0, 0.0)

    ## Execute Inverse transform
    fft_r = fft_c2r()

    ## Square in real space
    fft_r = (fft_r * M) ** 2

    ## Excute transform back to Fourier space
    ifft_c = fft_r2c(fft_r)

    ## Compute RHS using the compute convolution
    for i in range(k0 + 1, num_osc):
        phi[i] = (-i / np.absolute(cplx[i])) * np.real(np.exp(np.complex(0.0, -1.0) * np.angle(cplx[i])) * (ifft_c[i] / M))

    return phi, pert



@njit
def RKsolver(x, x_ext, RHS, dt):

    ## Stages
    rk1, rk1_ext = RHS(x, x_ext)
    rk2, rk2_ext = RHS(x + (0.5 * dt) * rk1, x_ext + (0.5 * dt) * rk1_ext)
    rk3, rk3_ext = RHS(x + (0.5 * dt) * rk2, x_ext + (0.5 * dt) * rk2_ext)
    rk4, rk4_ext = RHS(x + dt * rk3, x_ext + dt * rk3_ext)

    ## Update steps
    x     = x + (dt / 6.0) * (rk1 + 2.0 * rk2 + 2.0 * rk3 + rk4)
    x_ext = x_ext + (dt / 6.0) * (rk1_ext + 2.0 * rk2_ext + 2.0 * rk3_ext + rk4_ext) 

    return x, x_ext




@njit 
def compute_LEs(rates, run_sum, LEs, t, t0, numLEs):

    for i in range(numLEs):
        run_sum[i] += np.log(rates[i])
        LEs[i]     = run_sum[i] / (t)

    return LEs, run_sum




def print_update(m, t, m_iters, m_end, LEs, numLEs):

    if np.mod(m, (m_end * 0.1)) == 0:
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



@njit
def forward_dynamics(x, x_ext, RHS, dt, m_iters, m_end, m_trans, m_rev_trans, dof, numLEs):

    ## Allocate memory
    LEs     = np.zeros(dof)
    rates   = np.zeros(dof)
    run_sum = np.zeros(dof)

    ## CLV arrays
    R_tmp     = np.zeros((m_end - m_trans, dof, numLEs))
    Tang_vecs = np.zeros((m_end - m_trans - m_rev_trans, dof, numLEs))

    ## Output arrays
    x_out     = np.zeros((m_end, dof))
    x_ext_out = np.zeros((m_end, dof, numLEs))

    ## Iteration counters
    m     = 0
    iters = 1

    ## Begin algorithm
    while m < m_end:

        ## Integrate the system forward
        for i in range(m_iters):
        
            ## Call solver
            x, x_ext = RKsolver(x, x_ext, RHS, dt)

            ## Update iteration counter
            t     = iters * dt
            iters += 1


        ## Perform a QR decompostion of the tangent space
        Q, R = np.linalg.qr(x_ext)
        x_ext = Q

        ## Extrat growth rates - diagonals
        local_growth_rates = np.absolute(np.diag(R))

        ## Record system state
        x_out[m, :] = x
        x_ext_out[m, :] = x_ext

        
        ## If transient iterations have been reached - compute LEs
        if m >= m_trans:

            ## Compute the current LEs
            LEs, run_sum = compute_LEs(local_growth_rates, run_sum, LEs, t, (m_trans) * dt, numLEs)

            ## Print update
            # print_update(m, t, m_iters, m_end, LEs, numLEs)

            ## Save the QR data for the CLVs
            R_tmp[m - m_trans, :, :]  = R
            if m < m_end - m_rev_trans:
                Tang_vecs[m - m_trans, :, :] = Q

        ## Update for next iteration
        m += 1

    return x_out, x_ext_out, LEs, R_tmp, Tang_vecs


# @njit
def backward_dynamics(R_tmp, GS_vecs, m_rev_iters, m_rev_trans, dof, numLEs):

    ## Allocate memory
    CLVs = np.zeros((m_rev_iters - m_rev_trans, dof, numLEs))

    ## Create random upper triangular matrix - coeffiecients of the CLVs
    # C = np.array([[1.0, 0.5, 1/3], [0, 0.5, 1/3], [0, 0, 1/3]])
    C = np.triu(np.random.rand(dof, numLEs))
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

    print("Reverse Iters = {}".format(rev_iters))
    return CLVs


###########################
##  Main
###########################
if __name__ == '__main__':

    ## Lorenz setup parameters
    m_trans     = 10000
    m_avg       = 100000
    m_end       = m_avg + m_trans
    m_iters     = 100
    m_rev_trans = m_trans

    sigma  = 10.0
    rho    = 28.0
    beta   = 2.6666666666666667
    dof    = 3
    numLEs = dof
    t0     = 0.0
    t      = t0
    dt     = 0.001
    T      = dt * m_iters 


    ## Get initial conditions
    x, x_ext = initial_conditions(dof, "Lorenz63b")

    # x, x_ext = Lorenz_RHS(x, x_ext)

    # x, x_ext = RKsolver(x, x_ext, Lorenz_RHS, dt)


    x, x_ext, LEs, R_tmp, GS = forward_dynamics(x, x_ext, Lorenz_RHS, dt, m_iters, m_end, m_trans, m_rev_trans, dof, numLEs)

    print("LEs")
    print(LEs)

    print()
    print("x")
    print(x[m_trans - 1, :])  

    print()
    print("x_ext")
    print(x_ext[m_trans - 1, :])
    print(x_ext[-1, :])


    CLVs = backward_dynamics(R_tmp, GS, m_end - m_trans, m_rev_trans, dof, numLEs)

    print()
    print("CLVs")
    print(CLVs[0, :, :])
    print(CLVs[1, :, :])


    ## Plot CLV convergence
    t        = np.arange(m_end - m_trans - m_rev_trans)
    rate     = - t * np.absolute(LEs[0] - LEs[1])
    clv1_err = np.zeros((int(m_end - m_trans - m_rev_trans / 100), ))
    iters = 0
    for i in range(m_end - m_trans - m_rev_trans - 1, -1, -100):
        clv1_err[iters] = np.log(np.linalg.norm(CLVs[iters, :, 0] - CLVs[0, :, 0]))
        iters += 1

    plt.plot(rate, '-')
    plt.plot(clv1_err, '.-')
    plt.savefig("/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Testing_Functions/CLV" + "/CLV_Convergence.pdf")  
    plt.close()


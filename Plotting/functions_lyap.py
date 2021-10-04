import os
import sys
import h5py
import numpy as np
from scipy.linalg import subspace_angles
from numba import njit, jit, prange
import scipy.stats
import itertools


def open_file_lyap(a, n, k0, beta, u0, m_end, m_iter, trans, numLEs, input_dir):


    ## Get Filename
    filename       = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(n, k0, a, beta, u0, m_end * m_iter, m_end, m_iter, trans, numLEs)
    filename10     = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(n, k0, a, beta, u0, m_end * m_iter, m_end, m_iter, int(trans/10), numLEs)
    filename100    = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(n, k0, a, beta, u0, m_end * m_iter, m_end, m_iter, int(trans/100), numLEs)
    filename1000   = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(n, k0, a, beta, u0, m_end * m_iter, m_end, m_iter, int(trans/1000), numLEs)
    filename10000  = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(n, k0, a, beta, u0, m_end * m_iter, m_end, m_iter, int(trans/10000), numLEs)
    filename100000 = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(n, k0, a, beta, u0, m_end * m_iter, m_end, m_iter, int(trans/100000), numLEs)


    ## Open in current file
    if os.path.exists(input_dir + filename + '.h5'):
        print("a = {:0.3f} || Filename: {}".format(a, input_dir + filename + ".h5"))
        HDFfileData = h5py.File(input_dir + filename + '.h5', 'r')
    elif os.path.exists(input_dir + filename10 + '.h5'):
        print("a = {:0.3f} || Filename: {}".format(a, input_dir + filename10 + ".h5"))
        HDFfileData = h5py.File(input_dir + filename10 + '.h5', 'r')
    elif os.path.exists(input_dir + filename100 + '.h5'):
        print("a = {:0.3f} || Filename: {}".format(a, input_dir + filename100 + ".h5"))
        HDFfileData = h5py.File(input_dir + filename100 + '.h5', 'r')
    elif os.path.exists(input_dir + filename1000 + '.h5'):
        print("a = {:0.3f} || Filename: {}".format(a, input_dir + filename1000 + ".h5"))
        HDFfileData = h5py.File(input_dir + filename1000 + '.h5', 'r')
    elif os.path.exists(input_dir + filename10000 + '.h5'):
        print("a = {:0.3f} || Filename: {}".format(a, input_dir + filename10000 + ".h5"))
        HDFfileData = h5py.File(input_dir + filename10000 + '.h5', 'r')
    elif os.path.exists(input_dir + filename100000 + '.h5'):
        print("a = {:0.3f} || Filename: {}".format(a, input_dir + filename100000 + ".h5"))
        HDFfileData = h5py.File(input_dir + filename100000 + '.h5', 'r')
    else:
        print("File doesn't exist!...Alpha = {:.3f}".format(a))
        return -1


    return HDFfileData


def open_file_paper(N, k0, a, beta, u0, iters, m_end, m_iter, trans, numLEs, func_type, input_dir):
    
    HDFfileData = -1
    dof = int(N /2 - k0)
    
    ## Check if file exists and open
    if numLEs == 1:
        ## Create filename from data
        filename = input_dir + "/PAPER_LCEData_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{},{},{}]".format(N, k0, a, beta, u0, iters, m_end, m_iter)

        ## Check if file exists and open
        if os.path.exists(filename + "_TRANS[{}]_LEs[{}].h5".format(trans, numLEs)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]_LEs[{}].h5".format(trans, numLEs), 'r')
        elif os.path.exists(filename + "_TRANS[{}]_LEs[{}].h5".format(trans * 10, numLEs)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]_LEs[{}].h5".format(trans * 10, numLEs), 'r')
        elif os.path.exists(filename + "_TRANS[{}]_LEs[{}].h5".format(trans * 100, numLEs)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]_LEs[{}].h5".format(trans * 100, numLEs), 'r')
        elif os.path.exists(filename + "_TRANS[{}]_LEs[{}].h5".format(trans / 100, numLEs)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]_LEs[{}].h5".format(trans, numLEs), 'r')
        else: 
            print("File doesn't exist, check parameters!")
#             sys.exit()        
    else:
        ## Create filename from data
        filename = input_dir + "/PAPER_LCEData_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{},{},{}]".format(N, k0, a, beta, u0, iters, m_end, m_iter)

        if os.path.exists(filename + "_TRANS[{}].h5".format(trans)):
            HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans), 'r')
        elif os.path.exists(filename + "_TRANS[{}].h5".format(trans * 10)):
            HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans * 10), 'r')
        elif os.path.exists(filename + "_TRANS[{}].h5".format(trans / 10)):
            HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans / 10), 'r')
        elif os.path.exists(filename + "_TRANS[{}].h5".format(trans * 100)):
            HDFfileData = h5py.File(filename + "_TRANS[{}].h5".format(trans * 100), 'r')
        elif os.path.exists(filename + "_TRANS[{}]".format(trans) + "_LEs[{}].h5".format(dof)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]".format(trans) + "_LEs[{}].h5".format(dof), 'r')
        elif os.path.exists(filename + "_TRANS[{}]".format(trans * 10) + "_LEs[{}].h5".format(dof)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]".format(trans * 10) + "_LEs[{}].h5".format(dof), 'r')
        elif os.path.exists(filename + "_TRANS[{}]".format(trans / 10) + "_LEs[{}].h5".format(dof)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]".format(trans / 10) + "_LEs[{}].h5".format(dof), 'r')
        elif os.path.exists(filename + "_TRANS[{}]".format(trans * 100) + "_LEs[{}].h5".format(dof)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]".format(trans * 100) + "_LEs[{}].h5".format(dof), 'r')
        elif os.path.exists(filename + "_TRANS[{}]".format(trans * 1000) + "_LEs[{}].h5".format(dof)):
            HDFfileData = h5py.File(filename + "_TRANS[{}]".format(trans * 1000) + "_LEs[{}].h5".format(dof), 'r')
        else: 
            print("File doesn't exist, check parameters!")
#             sys.exit()

    return HDFfileData


def compute_kaplan_yorke(non_zero_spectrum, minindx):

    ## Counters
    lcesum = 0.0
    k_indx = int(0)
    kaplan_yorke = 0.0

    ## Loop over spectrum 
    for l in non_zero_spectrum:

        ## Check if sum is = 0
        if (lcesum + l) > 0.0:
            lcesum += l
            k_indx += 1
        else:
             break

    ## Compute Kaplan-Yorke dim
    if minindx == 0:
        kaplan_yorke = 0.0
    else:
        kaplan_yorke = k_indx + (lcesum / np.absolute(non_zero_spectrum[k_indx]))

    return kaplan_yorke


@njit
def compute_entropy(clv, dof):

    ## Entropy counter
    H = 0.0

    ## Loop over vector elements
    for i in range(dof):
        H += clv[i] * np.log(clv[i])

    return -H 

@njit
def compute_centroid(clv, dof, kmin):

    ## Centroid 
    C = 0.0

    ## Loop over vector elements
    for i in range(dof):
        C += clv[i] * (i + kmin)

    return C 

@njit
def compute_std(clv, cent, dof, kmin):

    ## Std Dev
    std = 0.0
    
    ## Compute std dev using associated probs
    for i in range(dof):
        std += (((i + kmin) - cent) ** 2) * clv[i]

    return np.sqrt(std)

@njit
def compute_chaotic_centroid(clv, dof, kmin, num_vecs):

    ## Centroid
    C = 0.0
    
    ## Loop over chaotic vectors
    for n in range(num_vecs):
        for i in range(dof):
            C += clv[i, n] * (i + kmin)

    return C / num_vecs

@njit
def compute_cumulative(data, ):

    c = np.zeros(data.shape[0])
    
    for i in range(data.shape[0]):
        for j in range(i):
            c[i] += data[j]
            
    return c


@njit
def compute_clv_stats_data(clv, a_k, num_tsteps, kmin, dof, numLEs):
    
    ## Memory Allocation
    v_k      = np.zeros((dof, dof))
    p_k      = np.zeros((dof, dof))
    v_k_proj = np.zeros((dof, dof))

    ## Translation Invariant Direction -> T
    T           = np.arange(2.0, float(dof + 2), 1.0)
    T_a_k       = T * a_k[kmin:]
    T_norm_sqr  = np.linalg.norm(T) ** 2
    T_enorm_sqr = np.linalg.norm(T_a_k) ** 2
    
    ## Loop over time
    for t in range(num_tsteps):

        ## Loop over vectors
        for j in range(numLEs):
            
            ## Square each component
            v_k[:, j] += np.square(clv[t, :, j])

            ## Compute the projection
            v_proj  = clv[t, :, j] - (T * (np.dot(clv[t, :, j], T))) / T_norm_sqr
            clv_a_k = clv[t, :, j] * a_k[kmin:]
            v_enorm = clv_a_k - (T_a_k * np.dot(clv_a_k, T_a_k)) / T_enorm_sqr
            
            ## Renormalize after projection
            v_proj     = v_proj / np.linalg.norm(v_proj)
            v_enorm = v_enorm / np.linalg.norm(v_enorm)
            
            ## Update running sum
            p_k[:, j]      += np.square(v_enorm)
            v_k_proj[:, j] += np.square(v_proj)
            
    ## Compute averages
    v_k       = v_k / num_tsteps
    p_k       = p_k / num_tsteps
    v_k_proj  = v_k_proj / num_tsteps
    

    return v_k, v_k_proj, p_k


@njit
def compute_max_clv_stats_data(clv, a_k, num_tsteps, kmin, dof):
    
    ## Memory Allocation
    v_k      = np.zeros((dof))
    p_k      = np.zeros((dof))
    v_k_proj = np.zeros((dof))

    ## Translation Invariant Direction -> T
    T           = np.arange(2.0, float(dof + 2), 1.0)
    T_a_k       = T * a_k[kmin:]
    T_norm_sqr  = np.linalg.norm(T) ** 2
    T_enorm_sqr = np.linalg.norm(T_a_k) ** 2
    
    ## Loop over time
    for t in range(num_tsteps):
            
        ## Square each component
        v_k[:] += np.square(clv[t, :])

        ## Compute the projection
        v_proj  = clv[t, :] - (T * (np.dot(clv[t, :], T))) / T_norm_sqr
        clv_a_k = clv[t, :] * a_k[kmin:]
        v_enorm = clv_a_k - (T_a_k * np.dot(clv_a_k, T_a_k)) / T_enorm_sqr
        
        ## Renormalize after projection
        v_proj  = v_proj / np.linalg.norm(v_proj)
        v_enorm = v_enorm / np.linalg.norm(v_enorm)
        
        ## Update running sum
        p_k[:]      += np.square(v_enorm)
        v_k_proj[:] += np.square(v_proj)
            
    ## Compute averages
    v_k      = v_k / num_tsteps
    p_k      = p_k / num_tsteps
    v_k_proj = v_k_proj / num_tsteps
    

    return v_k, v_k_proj, p_k
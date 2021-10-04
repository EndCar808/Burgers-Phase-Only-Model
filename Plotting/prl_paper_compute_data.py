######################
##	Library Imports ##
######################
import h5py
import sys
import os
import numpy as np
from scipy.linalg import subspace_angles
from numba import njit, jit, prange
import scipy.stats
import itertools


#######################
##	 Function Defs   ##
#######################
def open_file(N, k0, a, beta, u0, iters, m_end, m_iter, trans, numLEs, func_type):
    
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
        # filename = input_dir + "/LCEData_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]_ITERS[{},{},{}]".format(N, k0, a, beta, u0, iters, m_end, m_iter)

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


def open_file_lce_new(a, n, k0, beta, u0, iters, m_end, m_itr, trans, dof):

    ## Get Filename
    filename       = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(n, k0, a, beta, u0, iters, m_end, m_itr, trans, dof)
    filename10     = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(n, k0, a, beta, u0, iters, m_end, m_itr, int(trans/10), dof)
    filename100    = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(n, k0, a, beta, u0, iters, m_end, m_itr, int(trans/100), dof)
    filename1000   = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(n, k0, a, beta, u0, iters, m_end, m_itr, int(trans/1000), dof)
    filename10000  = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(n, k0, a, beta, u0, iters, m_end, m_itr, int(trans/10000), dof)
    filename100000 = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/LCEData_ITERS[{},{},{}]_TRANS[{}]_LEs[{}]".format(n, k0, a, beta, u0, iters, m_end, m_itr, int(trans/100000), dof)
    
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
def compute_cumulative(data):

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
    T              = np.arange(2.0, float(dof + 2), 1.0)
    T_a_k          = T * a_k[kmin:]
    T_norm_sqr     = np.linalg.norm(T) ** 2
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



######################
##	     MAIN       ##
######################
if __name__ == '__main__':

    ## ---------- Dataspace and Params
    N     = [128, 256, 512, 1024]
    alpha = np.arange(0.0, 2.51, 0.05)
    alpha_sm = np.array([0.0, 0.5, 1.0, 1.2, 1.5, 1.7, 2.0, 2.2, 2.5])
    k0    = 1
    beta  = 0.0
    iters = 400000
    m_end = 8000
    m_itr = 50
    trans = 1000000
    u0    = "RANDOM"

    ## ---------- Input and Output Dirs
    # input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/DataPRL_PAPER_DATA/Plots"
    input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Test"
    output_dir = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Snapshots/Paper"

    ## ---------- Memory Allocation
    ## LCEs
    deg_of_freedom   = np.zeros((len(N)))
    kurtosis         = np.zeros((len(N), len(alpha)))
    skewness         = np.zeros((len(N), len(alpha)))
    kaplan_yorke_dim = np.zeros((len(N), len(alpha)))
    max_entropy      = np.zeros((len(N), len(alpha)))
    ## CLVs
    max_entropy       = np.zeros((len(N), len(alpha)))
    max_entropy_proj  = np.zeros((len(N), len(alpha)))
    max_entropy_enorm = np.zeros((len(N), len(alpha)))
    max_mean_k        = np.zeros((len(N), len(alpha)))
    max_mean_k_proj   = np.zeros((len(N), len(alpha)))
    max_mean_k_enorm  = np.zeros((len(N), len(alpha)))
    pos_mean_k        = np.zeros((len(N), len(alpha)))
    pos_mean_k_proj   = np.zeros((len(N), len(alpha)))
    pos_mean_k_enorm  = np.zeros((len(N), len(alpha)))    
    quartiles         = np.zeros((2, len(N), len(alpha)))
    chaotic_quartiles = np.zeros((2, len(N), len(alpha)))

    max_clv_stats = np.zeros((len(N), len(alpha)))

    max_lyap_exp = np.zeros((len(N), len(alpha)))
    min_lyap_exp = np.zeros((len(N), len(alpha)))

    max_clv = np.zeros((len(N), len(alpha_sm)))
    
    lce_large = np.zeros((len(alpha), int(1024 / 2 + 1)))

    ## --------- Loop over Data
    for i, n in enumerate(N):

        ## Compute the available DOF
        deg_of_freedom[i] = n / 2 - k0

        if n == 1024:
            lce_large = np.zeros((len(alpha), int(n / 2 - k0)))

        for j, a in enumerate(alpha):

            print("N = {}, a = {:0.2f}".format(n, a))

            ## Open data file
            # HDFfileData = open_file(n, k0, a, beta, u0, iters, m_end, m_itr, 1000, int(n / 2 - k0), "max")
            HDFfileData = open_file_lce_new(a, n, k0, beta, u0, iters, m_end, m_itr, trans, int(n / 2 - k0))
            if HDFfileData == -1:
                # print("[ERROR] --- File Failed to Open -->> Skipping this alpha value [{}] for N = {}".format(a, n))
                continue

            ## ------- Read in Parameter Data
            amps    = HDFfileData['Amps'][:]
            kmin    = k0 + 1
            num_osc = amps.shape[0]
            dof     = num_osc - kmin


            ## ----------------- ##
            ## ------ LCEs ----- ##
            ## ----------------- ##
            ## Read in LCEs
            if 'LCEs' in list(HDFfileData.keys()):
                lce = HDFfileData['LCE'][-1, :]
            elif 'FinalLCE' in list(HDFfileData.keys()):
                lce = HDFfileData['FinalLCE'][:]
            else:
                print("[WARNING] --- No LCEs in Dataset -->> Skipping this alpha value [{}] for N = {}".format(a, n))
                continue

            ## LCE Stats
            kurtosis[i, j] = scipy.stats.kurtosis(lce)
            skewness[i, j] = scipy.stats.skew(lce)

            if n == 1024:
                lce_large[j, :] = lce
                   
            # find the zero mode
            minval  = np.amin(np.absolute(lce))
            minindx = np.where(np.absolute(lce) == minval)
            minindx_el,  = minindx

            # Extract the zero mode
            non_zero_spectrum = np.delete(lce, minindx, 0)

            # Find positive indices
            pos_indices = lce > 0 

            ## Kaplan-Yorke Dimension
            kaplan_yorke_dim[i, j] = compute_kaplan_yorke(non_zero_spectrum, minindx_el)

            ## Extract the maximal and minimal (positive) LE
            max_lyap_exp[i, j] = lce[0]
            min_lyap_exp[i, j] = lce[minindx_el - 1]

            ## ----------------- ##
            ## ------ CLVs ----- ##
            ## ----------------- ##
            ## Read in CLVs
            if 'CLVs' in list(HDFfileData.keys()):
                CLVs          = HDFfileData['CLVs']
                clv_dims      = CLVs.attrs['CLV_Dims']
                num_clv_steps = CLVs.shape[0]            
                clv           = np.reshape(CLVs, (CLVs.shape[0], dof, dof))

                ## Compute projected vectors
                v_k, v_k_proj, p_k = compute_clv_stats_data(clv, amps, num_clv_steps, kmin, dof, int(n / 2 - k0))

                ## Entropy
                max_entropy[i, j]       = compute_entropy(v_k[:, 0], dof)
                max_entropy_proj[i, j]  = compute_entropy(v_k_proj[:, 0], dof)
                max_entropy_enorm[i, j] = compute_entropy(p_k[:, 0], dof)
                ## Mean
                max_mean_k[i, j]       = compute_centroid(v_k[:, 0], dof, kmin)
                max_mean_k_proj[i, j]  = compute_centroid(v_k_proj[:, 0], dof, kmin)
                max_mean_k_enorm[i, j] = compute_centroid(p_k[:, 0], dof, kmin)
                pos_mean_k[i, j]       = compute_chaotic_centroid(v_k[:, pos_indices], dof, kmin, np.sum(pos_indices))
                pos_mean_k_proj[i, j]  = compute_chaotic_centroid(v_k_proj[:, pos_indices], dof, kmin, np.sum(pos_indices))
                pos_mean_k_enorm[i, j] = compute_chaotic_centroid(p_k[:, pos_indices], dof, kmin, np.sum(pos_indices))
                # Quartiles
                quartiles[0, i, j] = np.sum(compute_cumulative(p_k[:, 0]) < 0.25)
                quartiles[1, i, j] = np.sum(compute_cumulative(p_k[:, 0]) < 0.75)
            elif 'MaxCLVStats' in list(HDFfileData.keys()):
                max_clv_stats[i, j] = compute_centroid(HDFfileData['MaxCLVStats'][:], dof, kmin)
            else:
                print("[WARNING] --- No CLVs in Dataset -->> Skipping this alpha value [{:0.3f}] for N = {}".format(a, n))
                continue


    ## --------- Create & Write to File
    with h5py.File(output_dir + "/PaperData_new_data.hdf5", "w") as f:
        ## DOF
        f.create_dataset("DOF", data = deg_of_freedom)
        ## Kaplan-Yorke
        f.create_dataset("KaplanYorke", data = kaplan_yorke_dim)
        ## Save the minimal and maximal LE
        f.create_dataset("MaxLE", data = max_lyap_exp)
        f.create_dataset("MinLE", data = min_lyap_exp) 
        if 1024 in N:
            ## Save largest N LEs
            f.create_dataset("LargestLEs", data = lce_large)     
        if 'CLVs' in list(HDFfileData.keys()):
            ## Mean k
            f.create_dataset("MaxMeank",      data = max_mean_k)
            f.create_dataset("MaxMeankProj",  data = max_mean_k_proj)
            f.create_dataset("MaxMeankENorm", data = max_mean_k_enorm)
            ## Mean k - Chaotic directoins
            f.create_dataset("ChaoticMeank",      data = pos_mean_k)
            f.create_dataset("ChaoticMeankProj",  data = pos_mean_k_proj)
            f.create_dataset("ChaoticMeankENorm", data = pos_mean_k_enorm)
            ## Entropy
            f.create_dataset("MaxEntropy",      data = max_entropy)
            f.create_dataset("MaxEntropyProj",  data = max_entropy_proj)
            f.create_dataset("MaxEntropyENorm", data = max_entropy_enorm)
            ## Quartiles
            f.create_dataset("Quartiles", data = quartiles)
        elif 'MaxCLVStats' in list(HDFfileData.keys()):
            ## Max CLV stats
            f.create_dataset("MaxCLVMeanENorm", data = max_clv_stats)
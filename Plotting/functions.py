import sys
import os
import h5py
import numpy as np
import pyfftw as fftw
from numba import njit


def open_file(a, n, k0, beta, u0, iters, trans, input_dir):

    ## Get Filename
    filename       = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, trans)
    filename10     = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/10))
    filename100    = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/100))
    filename1000   = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/1000))
    filename10000  = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/10000))
    filename100000 = "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/SolverData_ITERS[{}]_TRANS[{}]".format(n, k0, a, beta, u0, iters, int(trans/100000))

    
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


def read_in_triads(file):

    if 'Triads' in list(file.keys()):
        triad  = file['Triads']
        # Reshape triads
        tdims     = triad.attrs['Triad_Dims']
        triads    = np.array(np.reshape(triad, np.append(triad.shape[0], tdims[0, :])))
    else: 
        print("Triads dataset does not exist!")

    return triads


@njit
def compute_current_triads(phases, kmin, kmax):
    # print("\n...Computing Triads...\n")

    ## Variables
    numTriads  = 0
    k3_range   = int(kmax - kmin + 1)
    k1_range   = int((kmax - kmin + 1) / 2)


    ## Create memory space
    triadphase = -10 * np.ones((k3_range, k1_range))
    triads     = -10 * np.ones((k3_range, k1_range))
    phaseOrder = np.complex(0.0, 0.0)
    
    ## Compute the triads
    for k in range(kmin, kmax + 1):
        for k1 in range(kmin, int(k/2) + 1):
            triadphase[k - kmin, k1 - kmin] = phases[k1] + phases[k - k1] - phases[k]
            triads[k - kmin, k1 - kmin]     = np.mod(triadphase[k - kmin, k1 - kmin], 2*np.pi)

            phaseOrder += np.exp(np.complex(0.0, 1.0)*triads[k - kmin, k1 - kmin])
            numTriads += 1
    
    # Compute Phase-Order params
    R   = np.absolute(phaseOrder / numTriads)
    Phi = np.angle(phaseOrder / numTriads)

    return triads, R, Phi

@njit
def compute_triads_all(phases, kmin, kmax):
    # print("\n...Computing Triads...\n")

    ## Variables
    numTriads  = 0
    k3_range   = int(kmax - kmin + 1)
    k1_range   = int((kmax - kmin + 1) / 2)
    time_steps = phases.shape[0]

    ## Create memory space
    triadphase = -10 * np.ones((k3_range, k1_range, time_steps))
    triads     = -10 * np.ones((k3_range, k1_range, time_steps))
    phaseOrder = np.complex(0.0, 0.0) * np.ones((time_steps))
    R          = np.zeros((time_steps))
    Phi        = np.zeros((time_steps))
    
    ## Compute the triads
    for k in range(kmin, kmax + 1):
        for k1 in range(kmin, int(k/2) + 1):
            triadphase[k - kmin, k1 - kmin, :] = phases[:, k1] + phases[:, k - k1] - phases[:, k]
            triads[k - kmin, k1 - kmin, :]     = np.mod(triadphase[k - kmin, k1 - kmin, :], 2*np.pi)

            phaseOrder[:] += np.exp(np.complex(0.0, 1.0)*triads[k - kmin, k1 - kmin, :])
            numTriads += 1
    
    # Compute Phase-Order params
    R[:]   = np.absolute(phaseOrder[:] / numTriads)
    Phi[:] = np.angle(phaseOrder[:] / numTriads)

    return triads, R, Phi


def extract_triads_all_k(triads, k_star, kmin, kmax):
    
    k_triads = []
    for k in range(kmin, kmax + 1):
        for k1 in range(kmin, int(k/2) + 1):
            if (k == k_star) or (k1 == k_star) or (k - k1 == k_star):
                k_triads.append(triads[k - kmin, k1 - kmin])
            else:
                continue

    return k_triads


def extract_triads_k(triads, k_star, kmin, kmax):
    
    k_triads = []
    for k in range(kmin, kmax + 1):
        for k1 in range(kmin, int(k/2) + 1):
            if (k == k_star) or (k1 == k_star) or (k - k1 == k_star):
                k_triads.append(triads[k - kmin, k1 - kmin, :])
            else:
                continue

    return k_triads



def compute_non_odered_triads_all(phases, kmin, kmax):
    
    triads_non_ordered = list()
    for t in range(phases.shape[0]):
        for k in range(kmin, kmax + 1):
            for k1 in range(-kmax + k, kmax + 1):
                if k - k1 <= -kmin:
                    if k1 <= -kmin and k1 >= -kmax:
                        triads_non_ordered.append(-phases[t, np.absolute(k1)] - phases[t, np.absolute(k - k1)] - phases[t, k])
                    if k1 >=  kmin and k1 <= kmax:
                        triads_non_ordered.append(phases[t, k1] - phases[t, np.absolute(k - k1)] - phases[t, k])
                elif k - k1 >= kmin:
                    if k1 <= -kmin and k1 >= -kmax:
                        triads_non_ordered.append(-phases[t, np.absolute(k1)] + phases[t, k - k1] - phases[t, k])
                    if k1 >=  kmin and k1 <= kmax:
                        triads_non_ordered.append(phases[t, k1] + phases[t, k - k1] - phases[t, k])


    return triads_non_odered


def compute_non_odered_triads_current(phases, t, kmin, kmax):
    
    triads_non_ordered = list()
    
    for k in range(kmin, kmax + 1):
        for k1 in range(-kmax + k, kmax + 1):
            if k - k1 <= -kmin:
                if k1 <= -kmin and k1 >= -kmax:
                    triads_non_ordered.append(-phases[t, np.absolute(k1)] - phases[t, np.absolute(k - k1)] - phases[t, k])
                if k1 >=  kmin and k1 <= kmax:
                    triads_non_ordered.append(phases[t, k1] - phases[t, np.absolute(k - k1)] - phases[t, k])
            elif k - k1 >= kmin:
                if k1 <= -kmin and k1 >= -kmax:
                    triads_non_ordered.append(-phases[t, np.absolute(k1)] + phases[t, k - k1] - phases[t, k])
                if k1 >=  kmin and k1 <= kmax:
                    triads_non_ordered.append(phases[t, k1] + phases[t, k - k1] - phases[t, k])


    return triads_non_odered


def compute_time(N, alpha, beta, k0, iters, trans):

    M = int(2 * N)
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

    ## Generate the time array
    start_time = trans * step
    end_time   = iters * step
    time = np.arange(start_time, end_time, step)

    ## Get transient iterations 
    trans_ratio = np.nanmax(np.absolute(rhs)) / np.nanmin(np.absolute(rhs))
    trans_mag   = np.ceil(np.log10(trans_ratio)) + 1
    trans_steps = 10 ** trans_ratio

    
    return time, step, trans_ratio



def compute_phases_rms(phases):
    phases_rms  = np.sqrt(np.mean(phases[:, :]**2, axis = 1))
    phases_prms = np.array([phases[i, :] / phases_rms[i] for i in range(phases.shape[0])])

    return phases_prms



## Real Space Data
def compute_realspace(amps, phases, N):
    print("\n...Creating Real Space Soln...\n")

    # Create full set of amps and phases
    amps_full   = np.append(amps[:], np.flipud(amps[1:-1]))
    phases_full = np.concatenate((phases[:, :], -np.fliplr(phases[:, 1:-1])), axis = 1)

    # Construct modes and realspace soln
    u_z = amps_full * np.exp(np.complex(0.0, 1.0) * phases_full)
    u   = np.real(np.fft.ifft(u_z, axis = 1, norm = 'ortho'))

    # Compute normalized realspace soln
    u_rms  = np.sqrt(np.mean(u[:, :]**2, axis = 1))
    u_urms = np.array([u[i, :] / u_rms[i] for i in range(u.shape[0])])

    x = np.arange(0, 2*np.pi, 2*np.pi/N)
    
    return u, u_urms, x, u_z




def compute_gradient(u_z, kmin, kmax):
    print("\nCreating Gradient\n")
    k            = np.concatenate((np.zeros((kmin)), np.arange(kmin, kmax + 1), -np.flip(np.arange(kmin, kmax)), np.zeros((kmin - 1))))
    grad_u_z     = np.complex(0.0, 1.0) * k * u_z
    du_x         = np.real(np.fft.ifft(grad_u_z, axis = 1, norm = 'ortho'))
    du_x_rms_tmp = np.sqrt(np.mean(du_x ** 2, axis = 1))
    du_x_rms     = np.array([du_x[i, :] / du_x_rms_tmp[i] for i in range(u_z.shape[0])])

    return du_x, du_x_rms
// Enda Carroll
// May 2020
// File including functions to perform the Benettin et al., algorithm
// for computing the Lyapunov spectrum of the Phase Only Burgers Equation


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <omp.h>
#include <gsl/gsl_cblas.h>
#include <lapacke.h>


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "utils.h"





// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
void po_rhs_extended(double* rhs, double* rhs_ext, fftw_complex* u_z, double* pert, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0) {


	///---------------
	/// RHS
	///---------------
	// Initialize variables
	int m = 2 * n;
	double norm_fac = 1.0 / (double) m;

	fftw_complex pre_fac;

	// Allocate temporary arrays
	double* u_tmp   = (double* ) malloc(m*sizeof(double));
	mem_chk(u_tmp, "u_tmp");
	double* jac_tmp = (double* ) malloc((num_osc - (k0 + 1)) * (num_osc - (k0 + 1)) *sizeof(double));
	mem_chk(jac_tmp, "jac_tmp");

	fftw_complex* conv    = (fftw_complex* ) fftw_malloc(num_osc * sizeof(fftw_complex));
	mem_chk(conv, "conv");
	fftw_complex* u_z_tmp = (fftw_complex* ) fftw_malloc((2 * num_osc - 1) * sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");
	


	// Write data to padded array
	for (int i = 0; i < (2 * num_osc - 1); ++i)	{
		if(i < num_osc){
			u_z_tmp[i] = u_z[i];
		} else {
			u_z_tmp[i] = 0.0 + 0.0*I;
		}
	}

	// transform back to Real Space
	fftw_execute_dft_c2r((*plan_c2r_pad), u_z_tmp, u_tmp);

	// multiplication in real space
	for (int i = 0; i < m; ++i)	{
		u_tmp[i] = pow(u_tmp[i], 2);
	}

	// transform forward to Fourier space
	fftw_execute_dft_r2c((*plan_r2c_pad), u_tmp, u_z_tmp);
	
	for (int k = 0; k < num_osc; ++k) {
		if (k <= k0) {
			rhs[k]  = 0.0;
			conv[k] = 0.0 + 0.0*I; 
		} else {
			pre_fac = (-I * kx[k]) / (2.0 * u_z[k]);
			conv[k] = u_z_tmp[k] * norm_fac;
			rhs[k]  = cimag(pre_fac * (conv[k]));
		}		
	}


	///---------------
	/// Extended RHS
	///---------------
	// calculate the jacobian
	int temp;
	int index;
	for (int kk = k0 + 1; kk < num_osc; ++kk) {
		temp = (kk - (k0 + 1)) * (num_osc - (k0 + 1));
		for (int kp = k0 + 1; kp < num_osc; ++kp) {
			index = temp + (kp - (k0 + 1));			
			if(kk == kp) { // Diagonal elements
				if(kk + kp <= num_osc - 1) {
					jac_tmp[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );
					jac_tmp[index] -= ((double) kk / 2.0) * cimag( conv[kp] / u_z[kk] );
				} else {
					jac_tmp[index] = 0.0;
					jac_tmp[index] -= ((double) kk / 2.0) * cimag( conv[kp] / u_z[kk] );
				} 
			} else { // Off diagonal elements
				if (kk + kp > num_osc - 1)	{
					if (kk - kp < -k0) {
						jac_tmp[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[abs(kk - kp)] )) / u_z[kk] );
					} else if (kk - kp > k0) {
						jac_tmp[index] = ((double) kk) * cimag( (u_z[kp] * u_z[kk - kp]) / u_z[kk] );
					} else {
						jac_tmp[index] = 0.0;
					}					
				} else {
					if (kk - kp < -k0) {
						jac_tmp[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[abs(kk - kp)] )) / u_z[kk] ) + ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );;
					} else if (kk - kp > k0) {
						jac_tmp[index] = ((double) kk) * cimag( (u_z[kp] * u_z[kk - kp]) / u_z[kk] ) + ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );;
					} else {
						jac_tmp[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );;
					}
				}
			}
		}
	}
		

	// Call matrix matrix multiplication - C = alpha*A*B + beta*C => rhs_ext = alpha*jac_tmp*pert + 0.0*C
	// variables setup
	double alpha = 1.0;     // prefactor of A*B
	double beta  = 0.0;     // prefactor of C
	int M = num_osc - (k0 + 1);    // no. of rows of A
	int N = num_osc - (k0 + 1);    // no. of cols of B
	int K = num_osc - (k0 + 1);    // no. of cols of A / rows of B
	int lda = num_osc - (k0 + 1);  // leading dim of A - length of elements between consecutive rows
	int ldb = num_osc - (k0 + 1);  // leading dim of B
	int ldc = num_osc - (k0 + 1);  // leading dim of C

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, jac_tmp, lda, pert, ldb, beta, rhs_ext, ldc);

	// Free tmp arrays
	free(jac_tmp);
	free(u_tmp);
	fftw_free(conv);
	fftw_free(u_z_tmp);
}




void jacobian(double* jac, fftw_complex* u_z, int n, int num_osc, int k0) {

	// Initialize temp vars
	int temp;
	int index;

	// initialize array for the convolution
	fftw_complex* conv = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));
	mem_chk(conv, "conv");
	
	// Call convolution for diagonal elements
	conv_direct(conv, u_z, num_osc, k0);

	// Loop through k and k'
	for (int kk = k0 + 1; kk < num_osc; ++kk) {
		temp = (kk - (k0 + 1)) * (num_osc - (k0 + 1));
		for (int kp = k0 + 1; kp < num_osc; ++kp) {
			index = temp + (kp - (k0 + 1));
			
			if(kk == kp) { // Diagonal elements
				if(kk + kp <= num_osc - 1) {
					jac[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );
					jac[index] -= ((double) kk / 2.0) * cimag( conv[kp] / u_z[kk] );
				} else {
					jac[index] = 0.0;
					jac[index] -= ((double) kk / 2.0) * cimag( conv[kp] / u_z[kk] );
				} 
			} else { // Off diagonal elements
				if (kk + kp > num_osc - 1)	{
					if (kk - kp < -k0) {
						jac[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[abs(kk - kp)] )) / u_z[kk] );
					} else if (kk - kp > k0) {
						jac[index] = ((double) kk) * cimag( (u_z[kp] * u_z[kk - kp]) / u_z[kk] );
					} else {
						jac[index] = 0.0;
					}					
				} else {
					if (kk - kp < -k0) {
						jac[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[abs(kk - kp)] )) / u_z[kk] ) + ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );;
					} else if (kk - kp > k0) {
						jac[index] = ((double) kk) * cimag( (u_z[kp] * u_z[kk - kp]) / u_z[kk] ) + ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );;
					} else {
						jac[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );;
					}
				}
			}
		}
	}

	///---------------
	/// Free temp memory
	///---------------
	fftw_free(conv);
}

double trace(fftw_complex* u_z, int n, int num_osc, int k0) {

	double tra;

	// initialize array for the convolution
	fftw_complex* conv = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));
	mem_chk(conv, "conv");
	
	// Call convolution for diagonal elements
	conv_direct(conv, u_z, num_osc, k0);

	tra = 0;
	for (int i = k0 + 1; i < num_osc; ++i)	{
		if(2*i <= num_osc - 1) {
			tra +=  ((double) i) * cimag( (u_z[i] * conj( u_z[2*i] )) / conj( u_z[i] ) );
			tra -= ((double) i / 2.0) * cimag( conv[i] / u_z[i] );
		} else {
			tra += 0.0;
			tra -= ((double) i / 2.0) * cimag( conv[i] / u_z[i] );
		} 
	}

	///---------------
	/// Free temp memory
	///---------------
	fftw_free(conv);

	return tra;
}

void modified_gs(double* q, double* r, int num_osc, int kmin) {

	int dim       = num_osc - kmin;
	double* r_tmp = (double* )malloc(sizeof(double) * dim * dim);
	double* a_tmp = (double* )malloc(sizeof(double) * dim * dim);

	// Write input to temporary matrix
	for (int i = 0; i < dim; ++i){
		for (int j = 0; j < dim; ++j) {
			a_tmp[i * dim + j ] = q[i * dim + j ] ;
		}
	}
	
	double norm;
	for (int i = 0; i < dim; ++i) {
		norm = 0.0;
		for (int k = 0; k < dim; ++k) {
				norm += a_tmp[k * dim + i]*a_tmp[k * dim + i];
		}
		r_tmp[i * dim + i] = sqrt(norm);

		// Write diagonal to ouotput
		r[i] = r_tmp[i * dim + i];

		for (int k = 0; k < dim; ++k) {
		 	q[k * dim + i]  =  a_tmp[k * dim + i] / r_tmp[i * dim + i]; 
		} 

		for (int j = i + 1; j < dim; ++j) {
			for (int k = 0; k < dim; ++k) {
				r_tmp[i * dim + j] += q[k * dim + i] * a_tmp[k * dim + j];
			}
			for (int k = 0; k < dim; ++k) {
				a_tmp[k * dim + j] -= r_tmp[i * dim + j] * q[k * dim + i]; 
			}
		}
	} 
}





void orthonormalize(double* pert, double* znorm, int num_osc, int kmin) {

	// Initialize vars
	int kdim = num_osc - kmin;

	// Initialize lapack vars
	lapack_int info;
	lapack_int m   = kdim;
	lapack_int n   = kdim;
	lapack_int lda = kdim;

	// Initialize the blas variables for dgemm
	double alpha = 1.0;     // prefactor of A*B
	double beta  = 0.0;     // prefactor of C
	int K = kdim;    // no. of cols of A / rows of B
	int ldb = kdim;  // leading dim of B
	int ldc = kdim;  // leading dim of C
	
	// Allocate temporary memory
	double* tau        = (double* )malloc(sizeof(double) * kdim);
	mem_chk(tau, "tau");	
	double* col_change =  (double* )malloc(sizeof(double) * (kdim) * (kdim));
	mem_chk(col_change, "col_change");
	double* rhs_pert   =  (double* )malloc(sizeof(double) * (kdim) * (kdim));
	mem_chk(rhs_pert, "rhs_pert");

	// Initialize col_change matrix
	// for (int i = 0; i < kdim; ++i) {
	// 	for (int j = 0; j < kdim; ++j) {
	// 		col_change[i * (kdim) + j] = 0.0; 
	// 		rhs_pert[i * (kdim) + j]   = pert[i * (kdim) + j];
	// 	}
	// }
	memset(col_change, 0.0, sizeof(double) * (kdim) * (kdim));
	memcpy(rhs_pert, pert, sizeof(double) * (kdim) * (kdim));

	///---------------
	/// Perform QR Fac
	///---------------
	info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, rhs_pert, lda, tau);


	// extract the diagonals of R
	for (int i = 0; i < kdim; ++i) {		
		if (rhs_pert[i * (kdim) + i] < 0) {
			col_change[i * (kdim) + i] = -1.0;
			znorm[i] = -1.0 *rhs_pert[i * (kdim) + i];
		} else {
			col_change[i * (kdim) + i] = 1.0;
			znorm[i] = rhs_pert[i * (kdim) + i];
		}
	}


	///---------------
	/// Form the Q matrix
	///---------------
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, m, n, rhs_pert, lda, tau);


   	// Correct the orientation of the Q matrix columns 
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, K, alpha, rhs_pert, lda, col_change, ldb, beta, pert, ldc);
	

	free(tau);
	free(col_change);
	free(rhs_pert);

}



void compute_lce_spectrum(int N, double a, double b, char* u0, int k0, int m_end, int m_iter) {

	// ------------------------------
	//  Variable Definitions
	// ------------------------------
	// Number of modes (incl 0 mode)
	int num_osc = (N / 2) + 1; 

	// padded array size
	int M = 2 * N;

	// Forcing wavenumber
	int kmin = k0 + 1;
	int kmax = num_osc - 1;

	// Triad phases array dims
	int k_range  = kmax - kmin + 1;
	int k1_range = (int)((kmax - kmin + 1)/ 2.0);

	// print update every x iterations
	int print_every = (m_end >= 10 ) ? (int)((double)m_end * 0.1) : 10;
	

	// Looping variables
	int tmp;
	int indx;

	// LCE variables
	double lce_sum;
	double dim_sum;
	int dim_indx;


	// ------------------------------
	//  Allocate memory
	// ------------------------------
	// Allocate mode related arrays
	int* kx  = (int* ) malloc(num_osc * sizeof(int));
	mem_chk(kx, "kx");
	double* amp   = (double* ) malloc(num_osc * sizeof(double));
	mem_chk(amp, "amp");
	double* phi   = (double* ) malloc(num_osc * sizeof(double));
	mem_chk(phi, "phi");
	double* u_pad = (double* ) malloc(M * sizeof(double));
	mem_chk(u_pad, "u_pad");
	fftw_complex* u_z = (fftw_complex* ) fftw_malloc(num_osc * sizeof(fftw_complex));
	mem_chk(u_z, "u_z");
	fftw_complex* u_z_pad = (fftw_complex* ) fftw_malloc((2 * num_osc - 1) * sizeof(fftw_complex));
	mem_chk(u_z_pad, "u_z_pad");

	// LCE Spectrum Arrays
	double* znorm    = (double* )malloc(sizeof(double) * (num_osc - kmin));	
	mem_chk(znorm, "znorm");
	double* lce      = (double* )malloc(sizeof(double) * (num_osc - kmin));
	mem_chk(lce, "lce");
	double* run_sum  = (double* )malloc(sizeof(double) * (num_osc - kmin));
	mem_chk(run_sum, "run_sum");
	#ifdef __LCE_ERROR
	double* lce_last = (double* )malloc(sizeof(double) * (num_osc - kmin));
	mem_chk(lce_last, "lce_last");
	double* abs_err  = (double* )malloc(sizeof(double) * (num_osc - kmin));
	mem_chk(abs_err, "abs_err");
	#endif

	#ifdef __TRIADS
	// Allocate array for triads
	double* triads = (double* )malloc(k_range * k1_range * sizeof(double));
	mem_chk(triads, "triads");
	
	// initialize triad array to handle empty elements
	for (int i = 0; i < k_range; ++i) {
		tmp = i * k1_range;
		for (int j = 0; j < k1_range; ++j) {
			indx = tmp + j;
			triads[indx] = -10.0;
		}
	}

	// Initilaize Phase Order peramter
	fftw_complex triad_phase_order;
	triad_phase_order = 0.0 + I * 0.0;
	#endif
	
	
	// ------------------------------
	// Runge-Kutta Variables / Arrays
	// ------------------------------
	// Define RK4 variables
	static double C2 = 0.5, A21 = 0.5, \
				  C3 = 0.5,           A32 = 0.5, \
				  C4 = 1.0,                      A43 = 1.0, \
				            B1 = 1.0/6.0, B2 = 1.0/3.0, B3 = 1.0/3.0, B4 = 1.0/6.0; 

	// Memory fot the four RHS evaluations in the stages 
	double* RK1, *RK2, *RK3, *RK4;
	RK1 = (double* )fftw_malloc(num_osc*sizeof(double));
	RK2 = (double* )fftw_malloc(num_osc*sizeof(double));
	RK3 = (double* )fftw_malloc(num_osc*sizeof(double));
	RK4 = (double* )fftw_malloc(num_osc*sizeof(double));
	mem_chk(RK1, "RK1");
	mem_chk(RK2, "RK2");
	mem_chk(RK3, "RK3");
	mem_chk(RK4, "RK4");

	// Temp array for intermediate modes
	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");

	// Memory for the four RHS evalutions for the perturbed system
	double* RK1_pert, *RK2_pert, *RK3_pert, *RK4_pert;
	RK1_pert = (double* )fftw_malloc((num_osc - kmin)*(num_osc - kmin)*sizeof(double));
	RK2_pert = (double* )fftw_malloc((num_osc - kmin)*(num_osc - kmin)*sizeof(double));
	RK3_pert = (double* )fftw_malloc((num_osc - kmin)*(num_osc - kmin)*sizeof(double));
	RK4_pert = (double* )fftw_malloc((num_osc - kmin)*(num_osc - kmin)*sizeof(double));
	mem_chk(RK1_pert, "RK1_pert");
	mem_chk(RK2_pert, "RK2_pert");
	mem_chk(RK3_pert, "RK3_pert");
	mem_chk(RK4_pert, "RK4_pert");

	// Memory for the solution to the perturbed system
	double* pert     = (double* ) malloc((num_osc - kmin)*(num_osc - kmin)*sizeof(double));
	mem_chk(pert, "pert");
	double* pert_tmp = (double* ) malloc((num_osc - kmin)*(num_osc - kmin)*sizeof(double));
	mem_chk(pert_tmp, "pert_tmp");

	// ------------------------------
	//  Create FFTW plans
	// ------------------------------
	// create fftw3 plans objects
	fftw_plan fftw_plan_r2c, fftw_plan_c2r;

	// create plans - ensure no overwriting - fill arrays after
	fftw_plan_r2c = fftw_plan_dft_r2c_1d(M, u_pad, u_z_pad, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(M, u_z_pad, u_pad, FFTW_PRESERVE_INPUT);




	// ------------------------------
	//  Get Initial Condition
	// ------------------------------
	// Set the initial condition of the perturb system to the identity matrix
	initial_conditions_lce(pert, phi, amp, u_z, kx, num_osc, k0, kmin, a, b, u0);

	if (num_osc <= 32) {
		for (int i = 0; i < num_osc; ++i) {
			printf("k[%d]: %d | a: %5.15lf   p: %5.16lf   | u_z[%d]: %5.16lf  %5.16lfI \n", i, kx[i], amp[i], phi[i], i, creal(amp[i] * cexp(I * phi[i])), cimag(amp[i] * cexp(I * phi[i])));
		}
		printf("\n\n");
	}


	// ------------------------------
	//  Get Timestep
	// ------------------------------
	double dt = get_timestep(amp, fftw_plan_c2r, fftw_plan_r2c, kx, N, num_osc, k0);

	#ifdef __TRANSIENTS
	#ifdef TRANS_STEPS
	int trans_iters = (int ) (TRANS_STEPS * (m_iter * m_end));	
	#else
	int trans_iters = get_transient_iters(amp, fftw_plan_c2r, fftw_plan_r2c, kx, N, num_osc, k0);
	#endif
	#else
	int trans_iters = 0;
	#endif

	// ------------------------------
	//  Setup Variables
	// ------------------------------
	// Saving variables
	int tot_m_save_steps = (int) m_end / SAVE_LCE_STEP;
	int tot_t_save_steps = (int) ((m_iter * m_end) / SAVE_DATA_STEP);	

	// LCE algorithm varibales
	int m = 1;
	#ifdef __TRANSIENTS
	#ifdef TRANS_STEPS
	int trans_m = (int ) (TRANS_STEPS * (m_end));	
	#else
	int trans_m = (int ) (trans_iters / m_iter);;
	#endif
	#else
	int trans_m = 0;
	#endif	
	

	// Solver time varibales 	
	double t0      = 0.0;
	double T       = t0 + m_iter * dt;	

	// printf("Tot Iters: %d - Trans Iters: %d ||| Mend: %d Trans m: %d ||| Trans Steps: %lf\n", (m_iter * m_end), trans_iters, m_end, trans_m, TRANS_STEPS);

	// ------------------------------
	//  HDF5 File Create
	// ------------------------------
	// Create the HDF5 file handle
	hid_t HDF_file_handle;


	// create hdf5 handle identifiers for hyperslabing the full evolution data
	hid_t HDF_file_space[4];
	hid_t HDF_data_set[4];
	hid_t HDF_mem_space[4];

	// get output file name
	char output_file_name[512];
	get_output_file_name(output_file_name, N, k0, a, b, u0, (m_iter * m_end), trans_iters);

	
	// open output file and create hyperslabbed datasets 
	open_output_create_slabbed_datasets_lce(&HDF_file_handle, output_file_name, HDF_file_space, HDF_data_set, HDF_mem_space, tot_t_save_steps, tot_m_save_steps, num_osc, k_range, k1_range, kmin);


	// Create arrays for time and phase order to save after algorithm is finished
	double* time_array      = (double* )malloc(sizeof(double) * (tot_t_save_steps + 1));
	mem_chk(time_array, "time_array");
	double* phase_order_R   = (double* )malloc(sizeof(double) * (tot_t_save_steps + 1));
	mem_chk(phase_order_R, "phase_order_R");
	double* phase_order_Phi = (double* )malloc(sizeof(double) * (tot_t_save_steps + 1));
	mem_chk(phase_order_Phi, "phase_order_Phi");

	// ------------------------------
	//  Write Initial Conditions to File
	// ------------------------------
	write_hyperslab_data_d(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], phi, "phi", num_osc, 0);


	#ifdef __TRIADS
	// compute triads for initial conditions
	triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
	
	// then write the current modes to this hyperslab
	write_hyperslab_data_d(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], triads, "triads", k_range * k1_range, 0);
	
	phase_order_R[0]   = cabs(triad_phase_order);
	phase_order_Phi[0] = carg(triad_phase_order);
	#endif

	// Write initial time
	#ifdef __TRANSIENTS
	time_array[0] = trans_iters * dt;
	#else
	time_array[0] = 0.0;
	#endif

	// ------------------------------
	//  Begin Algorithm
	// ------------------------------
	double t = 0.0;
	int iter = 1;
	int save_data_indx = 1;
	int save_lce_indx  = 1;
	while (m <= m_end) {

		// ------------------------------
		//  Integrate System Forward
		// ------------------------------
		for (int p = 0; p < m_iter; ++p) {

			// Construct the modes
			for (int i = 0; i < num_osc; ++i) {
				u_z_tmp[i] = amp[i] * cexp(I * phi[i]);
			}

			//////////////
			// STAGES
			//////////////
			/*---------- STAGE 1 ----------*/
			// find RHS first and then update stage
			po_rhs_extended(RK1, RK1_pert, u_z_tmp, pert, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);
			for (int i = 0; i < num_osc; ++i) {
				u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A21 * dt * RK1[i]));
					if (i < num_osc - kmin) {
						tmp = i * (num_osc - kmin);
						for (int j = 0; j < (num_osc - kmin); ++j) {
							indx = tmp + j;
							pert_tmp[indx] = pert[indx] + A21 * dt * RK1_pert[indx];
						}
					}
			}


			/*---------- STAGE 2 ----------*/
			// find RHS first and then update stage
			po_rhs_extended(RK2, RK2_pert, u_z_tmp, pert_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0) ;
			for (int i = 0; i < num_osc; ++i) {
				u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A32 * dt * RK2[i]));
					if (i < num_osc - kmin) {
						tmp = i * (num_osc - kmin);
						for (int j = 0; j < (num_osc - kmin); ++j) {
							indx = tmp + j;
							pert_tmp[indx] = pert[indx] + A21 * dt * RK2_pert[indx];
						}
					}
			}
			

			/*---------- STAGE 3 ----------*/
			// find RHS first and then update stage
			po_rhs_extended(RK3, RK3_pert, u_z_tmp, pert_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0) ;
			for (int i = 0; i < num_osc; ++i) {
				u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A43 * dt * RK3[i]));
					if (i < num_osc - kmin) {
						tmp = i * (num_osc - kmin);
						for (int j = 0; j < (num_osc - kmin); ++j) {
							indx = tmp + j;
							pert_tmp[indx] = pert[indx] + A43 * dt * RK3_pert[indx];
						}
					}
			}

			
			/*---------- STAGE 4 ----------*/
			// find RHS first and then update 
			po_rhs_extended(RK4, RK4_pert, u_z_tmp, pert_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0) ;

			
			//////////////
			// Update
			//////////////
			for (int i = 0; i < num_osc; ++i) {
				phi[i] = phi[i] + (dt * B1) * RK1[i] + (dt * B2) * RK2[i] + (dt * B3) * RK3[i] + (dt * B4) * RK4[i];  
					if (i < num_osc - kmin) {
						tmp = i * (num_osc - kmin);
						for (int j = 0; j < (num_osc - kmin); ++j) {
							indx = tmp + j;
							pert[indx] = pert[indx] + (dt * B1) * RK1_pert[indx] + (dt * B2) * RK2_pert[indx] + (dt * B3) * RK3_pert[indx] + (dt * B4) * RK4_pert[indx];  
						}
					}
			}			
			

			//////////////
			// Print to file
			//////////////
			if ((iter > trans_iters) && (iter % SAVE_DATA_STEP == 0)) {
				// Write phases
				write_hyperslab_data_d(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], phi, "phi", num_osc, save_data_indx);


				#ifdef __TRIADS
				// compute triads for initial conditions
				triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
				
				// write triads
				write_hyperslab_data_d(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], triads, "triads", k_range * k1_range, save_data_indx);

				// save phase order parameters
				phase_order_R[save_data_indx]   = cabs(triad_phase_order);
				phase_order_Phi[save_data_indx] = carg(triad_phase_order);
				#endif

				// save time
				time_array[save_data_indx] = iter * dt;

				
				// increment indx for next iteration
				save_data_indx += 1;
			}
			

			// increment
			t    = iter*dt;			
			iter += 1;			
		}		
		// ------------------------------
		//  End Integration
		// ------------------------------
		
		// ------------------------------
		//  Orthonormalize 
		// ------------------------------
		orthonormalize(pert, znorm, num_osc, kmin);

		
		// ------------------------------
		//  Compute LCEs & Write To File
		// ------------------------------
		if (m > trans_m) {
			lce_sum = 0.0;
			dim_sum = 0.0;
			dim_indx = 0;
			for (int i = 0; i < num_osc - kmin; ++i) {
				// Compute LCE
				run_sum[i] = run_sum[i] + log(znorm[i]);
				lce[i]     = run_sum[i] / (t - t0);

				#ifdef __LCE_ERROR
				// Compute Absolute Difference
				if (m > 1) {
					abs_err[i] = fabs(lce[i] - lce_last[i]);
				}
				// Save for next iteration
				lce_last[i] = lce[i];
				#endif
			}

			// then write the current LCEs to this hyperslab
			if (m % SAVE_LCE_STEP == 0) {			
				write_hyperslab_data_d(HDF_file_space[2], HDF_data_set[2], HDF_mem_space[2], lce, "lce", num_osc - kmin, save_lce_indx - 1);

				#ifdef __LCE_ERROR
				write_hyperslab_data_d(HDF_file_space[3], HDF_data_set[3], HDF_mem_space[3], abs_err, "lceError", num_osc - kmin, save_lce_indx - 1);
				#endif
				save_lce_indx += 1;
			}

			// Print update to screen
			if (m % print_every == 0) {
				lce_sum = 0.0;
				dim_sum = 0.0;
				dim_indx = 0;
				for (int i = 0; i < num_osc - kmin; ++i) {
					// Get spectrum sum
					lce_sum += lce[i];

					// Compute attractor dim
					if (dim_sum + lce[i] > 0) {
						dim_sum += lce[i];
						dim_indx += 1;
					}
					else {
						continue;
					}
				}
				printf("Iter: %d / %d | t: %5.6lf tsteps: %d | k0:%d alpha: %5.6lf beta: %5.6lf | Sum: %5.9lf | Dim: %5.9lf\n", m, m_end, t, m_end * m_iter, k0, a, b, lce_sum, (dim_indx + (dim_sum / fabs(lce[dim_indx]))));
				printf("k: \n");
				for (int j = 0; j < num_osc - kmin; ++j) {
					printf("%5.6lf ", lce[j]);
				}
				printf("\n\n");
			}
		}


		// ------------------------------
		//  Update For Next Iteration
		// ------------------------------
		T = T + m_iter * dt;
		m += 1;
		#ifdef __TRANSIENTS
		if (m - 1 == trans_m) {
			printf("\n\t!!Transient Iterations Complete!! - Iters: %d\n\n", iter - 1);
		}
		#endif
	}
	// ------------------------------
	//  Write 1D Arrays Using HDF5Lite
	// ------------------------------
	hid_t D1 = 1;
	hid_t D1dims[D1];

	// Write amplitudes
	D1dims[0] = num_osc;
	if ( (H5LTmake_dataset(HDF_file_handle, "Amps", D1, D1dims, H5T_NATIVE_DOUBLE, amp)) < 0){
		printf("\n\n!!Failed to make - Amps - Dataset!!\n\n");
	}

	// Wtie time
	D1dims[0] = tot_t_save_steps + 1;
	if ( (H5LTmake_dataset(HDF_file_handle, "Time", D1, D1dims, H5T_NATIVE_DOUBLE, time_array)) < 0) {
		printf("\n\n!!Failed to make - Time - Dataset!!\n\n");
	}

	#ifdef __TRIADS
	// Write Phase Order R
	D1dims[0] = tot_t_save_steps + 1;
	if ( (H5LTmake_dataset(HDF_file_handle, "PhaseOrderR", D1, D1dims, H5T_NATIVE_DOUBLE, phase_order_R)) < 0) {
		printf("\n\n!!Failed to make - PhaseOrderR - Dataset!!\n\n");
	}

	// Write Phase Order Phi
	D1dims[0] = tot_t_save_steps + 1;
	if ( (H5LTmake_dataset(HDF_file_handle, "PhaseOrderPhi", D1, D1dims, H5T_NATIVE_DOUBLE, phase_order_Phi)) < 0) {
		printf("\n\n!!Failed to make - PhaseOrderPhi - Dataset!!\n\n");
	}
	#endif


	// ------------------------------
	//  Clean Up & Exit
	// ------------------------------
	// Destroy DFT plans
	fftw_destroy_plan(fftw_plan_r2c);
	fftw_destroy_plan(fftw_plan_c2r);
	
	// Free memory
	#ifdef __TRIADS
	free(triads);
	free(phase_order_Phi);
	free(phase_order_R);
	#endif
	#ifdef __LCE_ERROR
	free(lce_last);
	free(abs_err);
	#endif
	free(kx);
	free(amp);
	free(phi);
	free(u_pad);
	free(znorm);
	free(lce);
	free(run_sum);
	free(time_array);
	free(RK1);
	free(RK2);
	free(RK3);
	free(RK4);
	free(RK1_pert);
	free(RK2_pert);
	free(RK3_pert);
	free(RK4_pert);
	free(pert);
	free(pert_tmp);
	fftw_free(u_z);
	fftw_free(u_z_tmp);
	fftw_free(u_z_pad);
	

	// // Close HDF5 handles
	#ifdef __TRIADS
	H5Sclose( HDF_mem_space[1] );
	H5Dclose( HDF_data_set[1] );
	H5Sclose( HDF_file_space[1] );
	#endif
	#ifdef __LCE_ERROR
	H5Sclose( HDF_mem_space[3] );
	H5Dclose( HDF_data_set[3] );
	H5Sclose( HDF_file_space[3] );
	#endif
	H5Sclose( HDF_mem_space[0] );
	H5Dclose( HDF_data_set[0] );
	H5Sclose( HDF_file_space[0] );
	H5Sclose( HDF_mem_space[2] );
	H5Dclose( HDF_data_set[2] );
	H5Sclose( HDF_file_space[2] );

	// Close output file
	H5Fclose(HDF_file_handle);
}
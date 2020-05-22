// Enda Carroll
// May 2020
// File including functions to perform a pseudospectral solving scheme 
// for a Phase Only model of the 1D Burgers equation


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <fftw3.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <omp.h>
#include <gsl/gsl_cblas.h>


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "po_data_types.h"
#include "utils.h"



// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Function to generate the initial condition for the solver and to initialize the wavenumber array
 * 
 * @param u          Array to store the initial condition in real space
 * @param u_z        Array to store the DFT of the initial conditon
 * @param real2compl FFTW plan to  perform DFT of the initial condiition - forward transform
 * @param k          Array to store the wavenumbers
 * @param dx         Value of increment in space
 * @param N          Value of the number of modes in the system
 */
void initial_condition(double* phi, double* amp, int* kx, int num_osc, int k0, int cutoff, double a, double b) {

	// set the seed for the random number generator
	srand(123456789);

	for (int i = 0; i < num_osc; ++i) {

		// fill the wavenumbers array
		kx[i] = i;

		// fill amp and phi arrays
		if(i <= k0) {
			amp[i] = 0.0;
			phi[i] = 0.0;
		} else {
			amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
			phi[i] = M_PI/2.0;	
			// phi[i] = M_PI*( (double) rand() / (double) RAND_MAX);	
		}
	}
}


void min(double* a, int n, int k0, double* min_val) {

	double min = fabs(a[k0 + 1]);

	for (int i = k0 + 1; i < n; ++i)	{
		if (fabs(a[i]) < min) {
			min = fabs(a[i]);
		}
	}
	*min_val = min;
}

void max(double* a, int n, int k0, double* max_val) {

	double max = fabs(a[k0 + 1]);

	for (int i = k0 + 1; i < n; ++i)	{
		if (fabs(a[i]) > max) {
			max = fabs(a[i]);
		}
	}
	*max_val = max;
}



void conv_2N_pad(fftw_complex* convo, fftw_complex* uz, fftw_plan *fftw_plan_r2c_ptr, fftw_plan *fftw_plan_c2r_ptr, int n, int num_osc, int k0) {

	// Padded resolution
	int m = 2*n;

	// Normalization factor
	double norm_fact = 1.0 / (double) m;

	// Allocate temporary arrays
	double* u_tmp;
	u_tmp = (double*)malloc(m*sizeof(double));
	fftw_complex* uz_pad;
	uz_pad = (fftw_complex*)malloc(2*num_osc*sizeof(fftw_complex));
	
	// write input data to padded array
	for (int i = 0; i < 2*num_osc; ++i)	{
		if(i < num_osc){
			uz_pad[i] = uz[i];
		} else {
			uz_pad[i] = 0.0 + 0.0*I;
		}
	}

	// // transform back to Real Space
	fftw_execute_dft_c2r((*fftw_plan_c2r_ptr), uz_pad, u_tmp);

	// // square
	for (int i = 0; i < m; ++i)	{
		u_tmp[i] = pow(u_tmp[i], 2);
	}

	// here we move the derivative from real to spectral space so that the derivatives can be returned in spectral space
	fftw_execute_dft_r2c((*fftw_plan_r2c_ptr), u_tmp, uz_pad);

	// // normalize
	for (int i = 0; i < num_osc; ++i)	{
		if (i <= k0) {
			convo[i] = 0.0 + 0.0*I;
		} else {
			convo[i] = uz_pad[i]*(norm_fact);
		}		
	}

	///---------------
	/// Free temp memory
	///---------------
	free(u_tmp);
	fftw_free(uz_pad);
}


void conv_23(fftw_complex* convo, fftw_complex* uz, fftw_plan *fftw_plan_r2c_ptr_23, fftw_plan *fftw_plan_c2r_ptr_23, int n, int kmax, int k0) {

	double norm_fact = 1/((double) n);

	double* u_tmp;
	u_tmp = (double* )malloc(n*sizeof(double));

	// transform back to Real Space
	fftw_execute_dft_c2r((*fftw_plan_c2r_ptr_23), uz, u_tmp);

	for (int i = 0; i < n; ++i)	{
		u_tmp[i] = pow(u_tmp[i], 2);
	}

	// here we move the derivative from real to spectral space so that the derivatives can be returned in spectral space
	fftw_execute_dft_r2c((*fftw_plan_r2c_ptr_23), u_tmp, convo);

	// normalize
	for (int i = 0; i < kmax; ++i)	{
		if(i <= k0)	{
			convo[i] = 0.0 + 0.0*I;
		} else {
			convo[i] *= norm_fact;	
		}
	}

	// apply filter mask
	for (int i = kmax; i < n; ++i)	{
		convo[i] = 0.0 + 0.0*I; 
	}


	///---------------
	/// Free temp memory
	///---------------
	free(u_tmp);
}


void conv_direct(fftw_complex* convo, fftw_complex* u_z, int n, int k0) {
	
	// Set the 0 to k0 modes to 0;
	for (int i = 0; i <= k0; ++i) {
		convo[0] = 0.0 + 0.0*I;
	}
	
	// Compute the convolution on the remaining wavenumbers
	int k1;
	for (int kk = k0 + 1; kk < n; ++kk)	{
		for (int k_1 = 1 + kk; k_1 < 2*n; ++k_1)	{
			// Get correct k1 value
			if(k_1 < n) {
				k1 = -n + k_1;
			} else {
				k1 = k_1 - n;
			}
			if (k1 < 0) {
				convo[kk] += conj(u_z[abs(k1)])*u_z[kk - k1]; 	
			} else if (kk - k1 < 0) {
				convo[kk] += u_z[k1]*conj(u_z[abs(kk - k1)]); 
			} else {
				convo[kk] += u_z[k1]*u_z[kk - k1];
			}			
		}
	}
}


void po_rhs(double* rhs, fftw_complex* u_z, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0) {

	// Initialize variables
	int m = 2 * n;
	double norm_fac = 1.0 / (double) m;

	fftw_complex pre_fac;

	// Allocate temporary arrays
	double* u_tmp;
	u_tmp = (double* ) malloc(m*sizeof(double));

	fftw_complex* u_z_tmp;
	u_z_tmp = (fftw_complex* ) fftw_malloc(2*num_osc*sizeof(fftw_complex));

	///---------------
	/// Convolution
	///---------------
	// Write data to padded array
	for (int i = 0; i < 2*num_osc; ++i)	{
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

	///---------------
	/// RHS
	///---------------
	for (int k = 0; k < num_osc; ++k) {
		if (k <= k0) {
			rhs[k] = 0.0;
		} else {
			pre_fac = (-I * kx[k]) / (2.0 * u_z[k]);
			rhs[k]  = cimag( pre_fac* (u_z_tmp[k] * norm_fac) );
		}		
	}

	///---------------
	/// Free temp memory
	///---------------
	free(u_tmp);
	fftw_free(u_z_tmp);
}


void po_rhs_extended(double* rhs, double* rhs_ext, fftw_complex* u_z, double* pert, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0) {


	///---------------
	/// RHS
	///---------------
	// Initialize variables
	int m = 2 * n;
	double norm_fac = 1.0 / (double) m;

	fftw_complex pre_fac;

	// Allocate temporary arrays
	double* u_tmp;
	u_tmp = (double* ) malloc(m*sizeof(double));

	double* jac_tmp;
	jac_tmp = (double* ) malloc((num_osc - (k0 + 1))*(num_osc - (k0 + 1))*sizeof(double));

	fftw_complex* u_z_tmp;
	u_z_tmp = (fftw_complex* ) fftw_malloc(2*num_osc*sizeof(fftw_complex));

	fftw_complex* conv;
	conv = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));

	// Write data to padded array
	for (int i = 0; i < 2*num_osc; ++i)	{
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

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, beta, jac_tmp, lda, pert, ldb, alpha, rhs_ext, ldc);


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
	fftw_complex* conv;
	conv = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));
	
	// Call convolution for diagonal elements
	conv_direct(conv, u_z, n, k0);

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
	fftw_complex* conv;
	conv = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));
	
	// Call convolution for diagonal elements
	conv_direct(conv, u_z, n, k0);

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

void triad_phases(double* triads, fftw_complex* phase_order, double* phi, int kmin, int kmax) {

	int num_triads;
	int tmp;
	int indx;

	double phase_val;

	fftw_complex phase_order_tmp;

	num_triads      = 0;
	phase_order_tmp = 0.0 + 0.0 * I;
	for (int k = kmin; k <= kmax; ++k) {
		tmp = (k - kmin) * (int) ((kmax - kmin + 1) / 2.0);
		for (int k1 = kmin; k1 <= (int) (k / 2.0); ++k1)	{
			indx = tmp + (k1 - kmin);

			// find the triad value
			phase_val = phi[k1] + phi[k - k1] - phi[k];

			// store the triad value
			triads[indx] = fmod(phase_val, 2*M_PI);

			// update phase order param
			phase_order_tmp += cexp(I * phase_val);
			
			// increment triad counter
			num_triads++;			
		}
	}

	// normalize phase order parameter
	*phase_order =  phase_order_tmp / (double) num_triads;
}


double get_timestep(double* amps, fftw_plan plan_c2r, fftw_plan plan_r2c, int* kx, int n, int num_osc, int k0) {

	// Initialize vars
	double dt;
	double max_val;

	// Initialize temp memory
	double* tmp_rhs;
	tmp_rhs = (double* ) fftw_malloc(num_osc*sizeof(double));

	fftw_complex* u_z_tmp;
	u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));

	// Create modes for RHS evaluation
	for (int i = 0; i < num_osc; ++i) {
		if (i <= k0) {
			u_z_tmp[i] = 0.0 + 0.0*I;
		} else {
			u_z_tmp[i] = amps[i] * exp(I * 0.0);
		}
	}

	// Call the RHS
	po_rhs(tmp_rhs, u_z_tmp, &plan_c2r, &plan_r2c, kx, n, num_osc, k0);

	// Find  the fastest moving oscillator
	max(tmp_rhs, num_osc, k0, &max_val);

	// Get timestep
	dt = 1.0 / max_val;

	free(tmp_rhs);
	fftw_free(u_z_tmp);
	
	return dt;
}


int get_transient_iters(double* amps, fftw_plan plan_c2r, fftw_plan plan_r2c, int* kx, int n, int num_osc, int k0) {

	// Initialize vars
	int trans_iters;
	double min_val;
	double max_val;
	double trans_ratio;
	int trans_mag;

	// Initialize temp memory
	double* tmp_rhs;
	tmp_rhs = (double* ) fftw_malloc(num_osc*sizeof(double));

	fftw_complex* u_z_tmp;
	u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));

	// Create modes for RHS evaluation
	for (int i = 0; i < num_osc; ++i) {
		if (i <= k0) {
			u_z_tmp[i] = 0.0 + 0.0*I;
		} else {
			u_z_tmp[i] = amps[i] * exp(I * 0.0);
		}
	}

	// Call the RHS
	po_rhs(tmp_rhs, u_z_tmp, &plan_c2r, &plan_r2c, kx, n, num_osc, k0);

	// Find  the fastest moving oscillator
	max(tmp_rhs, num_osc, k0, &max_val);

	// Find the slowest moving oscillator
	min(tmp_rhs, num_osc, k0, &min_val);

	// find the magnitude
	trans_ratio = max_val / min_val;
	trans_mag   = ceil(log10(trans_ratio)) + 1;

	// get the no. of iterations
	trans_iters = pow(10, trans_mag);

	return trans_iters;
}



double system_energy(fftw_complex* u_z, int N) {

	double sys_energy;

	// initialize the energy sum
	sys_energy = u_z[0]*conj(u_z[0]); // 0th mode does not have a conj by definion

	// loop over modes and find the total energy
	for (int i = 1; i < N/2 + 1; ++i)	{
		sys_energy += 2.0*u_z[i]*conj(u_z[i]);   // account for the Reality condition u_{-k} = - u_{k}
	}

	return sys_energy;
}


double system_enstrophy(fftw_complex* u_z, int* k, int N) {

	double sys_enstrophy;

	// initialize enstrophy sum
	sys_enstrophy = cabs(I*k[0]*u_z[0])*cabs(I*k[0]*u_z[0]);

	// loop over modes
	for (int i = 1; i < N/2 + 1; ++i) {
		sys_enstrophy += cabs(I*k[0]*u_z[0])*cabs(I*k[0]*u_z[0]);
	}

	return sys_enstrophy;
}
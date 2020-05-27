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
#include "utils.h"
#include "solver.h"




// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
void initial_conditions_lce(double* pert, double* phi, double* amp, int* kx, int num_osc, int k0, int kmin, double a, double b) {

	int tmp;
	int indx;

	// Spectrum cutoff
	double cutoff = ((double) num_osc - 1.0) / 2.0;

	// set the seed for the random number generator
	srand(123456789);

	// Fill phases, amps and wavenumbers
	for (int i = 0; i < num_osc; ++i) {

		// fill the wavenumbers array
		kx[i] = (int) i;

		// fill amp and phi arrays
		if(i <= k0) {
			amp[i] = 0.0;
			phi[i] = 0.0;
		} else {
			amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
			phi[i] = M_PI/2.0 * (1 + 1e-10 * pow(i, 0.9));	
			// phi[i] = M_PI*( (double) rand() / (double) RAND_MAX);	
		}

		// fill the perturbation array
		if (i < num_osc - kmin) {
			tmp = i * (num_osc - kmin);
			for (int j = 0; j < num_osc  - kmin; ++j)
			{
				indx = tmp + j;
				if (i == j) {
					pert[indx] = 1.0;
				} else {
					pert[indx] = 0.0;
				}
			}
		}
	}

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
	double* u_tmp   = (double* ) malloc(m*sizeof(double));
	double* jac_tmp = (double* ) malloc((num_osc - (k0 + 1)) * (num_osc - (k0 + 1)) *sizeof(double));

	fftw_complex* conv    = (fftw_complex* ) fftw_malloc(num_osc * sizeof(fftw_complex));
	fftw_complex* u_z_tmp = (fftw_complex* ) fftw_malloc(2 * num_osc * sizeof(fftw_complex));
	


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
	double* col_change =  (double* )malloc(sizeof(double) * (kdim) * (kdim));
	double* rhs_pert   =  (double* )malloc(sizeof(double) * (kdim) * (kdim));

	// Initialize col_change matrix
	for (int i = 0; i < kdim; ++i) {
		for (int j = 0; j < kdim; ++j) {
			col_change[i * (kdim) + j] = 0.0; 
			rhs_pert[i * (kdim) + j]   = pert[i * (kdim) + j];
		}
	}

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
		// printf("r[%d]: %20.15lf\n", i, znorm[i]);
		
	}
	// printf("\n\n");


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

void open_output_create_slabbed_datasets_lce(hid_t* file_handle, char* output_file_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, int num_t_steps, int num_m_steps, int num_osc, int k_range, int k1_range, int kmin) {

	// ------------------------------
	//  Create file
	// ------------------------------
	
	// create datafile - H5F_ACC_TRUNC overwrites file if it exists already
	*file_handle = H5Fcreate(output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);


	// ------------------------------
	//  Create datasets with hyperslabing
	// ------------------------------
	//
	//---------- PHASES -----------//
	//
	// create hdf5 dimension arrays for creating the hyperslabs
	static const int dimensions = 2;
	hsize_t dims[dimensions];      // array to hold dims of full evolution data
	hsize_t maxdims[dimensions];   // array to hold max dims of full evolution data
	hsize_t chunkdims[dimensions]; // array to hold dims of the hyperslab chunks

	// initialize the hyperslab arrays
	dims[0]      = num_t_steps + 1;             // number of timesteps
	dims[1]      = num_osc;                 // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = num_osc;                 // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = num_osc;                 // 1D chunk of size number of modes

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[0] = H5Screate_simple(dimensions, dims, maxdims);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist;
	plist = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist, dimensions, chunkdims);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[0] = H5Dcreate(*file_handle, "Phases", H5T_NATIVE_DOUBLE, file_space[0], H5P_DEFAULT, plist, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims[0] = 1;
	dims[1] = num_osc;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[0] = H5Screate_simple(dimensions, dims, NULL);

	H5Pclose(plist);

	//
	//---------- TRIADS -----------//
	//
	// create hdf5 dimension arrays for creating the hyperslabs
	static const int dim2 = 2;
	hsize_t dims2[dim2];      // array to hold dims of full evolution data
	hsize_t maxdims2[dim2];   // array to hold max dims of full evolution data
	hsize_t chunkdims2[dim2]; // array to hold dims of the hyperslab chunks

	// initialize the hyperslab arrays
	dims2[0]      = num_t_steps + 1;             // number of timesteps + initial condition
	dims2[1]      = k_range * k1_range;      // size of triads array
	maxdims2[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims2[1]   = k_range * k1_range;      // size of triads array
	chunkdims2[0] = 1;                       // 1D chunk to be saved 
	chunkdims2[1] = k_range*k1_range;         // size of triad array

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[1] = H5Screate_simple(dim2, dims2, maxdims2);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist2;
	plist2 = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist2, dim2, chunkdims2);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[1] = H5Dcreate(*file_handle, "Triads", H5T_NATIVE_DOUBLE, file_space[1], H5P_DEFAULT, plist2, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims2[0] = 1;
	dims2[1] = k_range*k1_range;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[1] = H5Screate_simple(dim2, dims2, NULL);

	// Create attribute data for the triad dimensions
	hid_t triads_attr, triads_attr_space;

	hsize_t adims[2];
	adims[0] = 1;
	adims[1] = 2;

	triads_attr_space = H5Screate_simple (2, adims, NULL);

	triads_attr = H5Acreate(data_set[1], "Triad_Dims", H5T_NATIVE_INT, triads_attr_space, H5P_DEFAULT, H5P_DEFAULT);

	int triads_dims[2];
	triads_dims[0] = k_range;
	triads_dims[1] = k1_range;

    herr_t status = H5Awrite(triads_attr, H5T_NATIVE_INT, triads_dims);

	// close the created property list
	status = H5Aclose(triads_attr);
    status = H5Sclose(triads_attr_space);
	status = H5Pclose(plist2);



	//---------- LCE -----------//
	//
	// initialize the hyperslab arrays
	dims[0]      = num_m_steps;             // number of timesteps
	dims[1]      = num_osc - kmin;          // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = num_osc - kmin;          // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = num_osc - kmin;          // 1D chunk of size number of modes

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[2] = H5Screate_simple(dimensions, dims, maxdims);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist3;
	plist3 = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist3, dimensions, chunkdims);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[2] = H5Dcreate(*file_handle, "LCE", H5T_NATIVE_DOUBLE, file_space[2], H5P_DEFAULT, plist3, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims[0] = 1;
	dims[1] = num_osc - kmin;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[2] = H5Screate_simple(dimensions, dims, NULL);

	H5Pclose(plist3);


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

	// ------------------------------
	//  Allocate memory
	// ------------------------------
	// Allocate mode related arrays
	int* kx  = (int* ) malloc(num_osc * sizeof(int));
	double* amp   = (double* ) malloc(num_osc * sizeof(double));
	double* phi   = (double* ) malloc(num_osc * sizeof(double));
	double* u_pad = (double* ) malloc(M * sizeof(double));
	fftw_complex* u_z = (fftw_complex* ) fftw_malloc(num_osc * sizeof(fftw_complex));
	fftw_complex* u_z_pad = (fftw_complex* ) fftw_malloc(2 * num_osc * sizeof(fftw_complex));

	// LCE Spectrum Arrays
	double* znorm   = (double* )malloc(sizeof(double) * (num_osc - kmin));	
	double* lce     = (double* )malloc(sizeof(double) * (num_osc - kmin));
	double* run_sum = (double* )malloc(sizeof(double) * (num_osc - kmin));

	
	// Allocate array for triads
	double* triads = (double* )malloc(k_range * k1_range * sizeof(double));
	
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

	// Temp array for intermediate modes
	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));

	// Memory for the four RHS evalutions for the perturbed system
	double* RK1_pert, *RK2_pert, *RK3_pert, *RK4_pert;
	RK1_pert = (double* )fftw_malloc((num_osc - kmin)*(num_osc - kmin)*sizeof(double));
	RK2_pert = (double* )fftw_malloc((num_osc - kmin)*(num_osc - kmin)*sizeof(double));
	RK3_pert = (double* )fftw_malloc((num_osc - kmin)*(num_osc - kmin)*sizeof(double));
	RK4_pert = (double* )fftw_malloc((num_osc - kmin)*(num_osc - kmin)*sizeof(double));

	// Memory for the solution to the perturbed system
	double* pert     = (double* ) malloc((num_osc - kmin)*(num_osc - kmin)*sizeof(double));
	double* pert_tmp = (double* ) malloc((num_osc - kmin)*(num_osc - kmin)*sizeof(double));


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
	initial_conditions_lce(pert, phi, amp, kx, num_osc, k0, kmin, a, b);

	for (int i = 0; i < num_osc; ++i) {
		printf("k[%d]: %d | a: %5.15lf   p: %5.16lf   | u_z[%d]: %5.16lf  %5.16lfI \n", i, kx[i], amp[i], phi[i], i, creal(amp[i] * cexp(I * phi[i])), cimag(amp[i] * cexp(I * phi[i])));
	}
	printf("\n\n");

	// print_array_2d_d(pert, "pert", num_osc - kmin, num_osc - kmin);

	// ------------------------------
	//  Get Timestep
	// ------------------------------
	double dt;
	dt = get_timestep(amp, fftw_plan_c2r, fftw_plan_r2c, kx, N, num_osc, k0);


	// ------------------------------
	//  Setup Variables
	// ------------------------------
	// LCE algorithm varibales
	int m         = 1;
	int save_step = 1;

	// Solver time varibales 
	int tot_tsteps = (int) ((m_iter * m_end) / save_step);
	
	double t0      = 0.0;
	double T       = t0 + m_iter * dt;	



	// ------------------------------
	//  HDF5 File Create
	// ------------------------------
	// Create the HDF5 file handle
	hid_t HDF_file_handle;


	// create hdf5 handle identifiers for hyperslabing the full evolution data
	hid_t HDF_file_space[3];
	hid_t HDF_data_set[3];
	hid_t HDF_mem_space[3];

	// // define filename - const because it doesnt change
	char output_file_name[128] = "../Data/Output/LCE_Runtime_Data";
	char output_file_data[128];

	// form the filename of the output file
	sprintf(output_file_data,  "_N[%d]_k0[%d]_ALPHA[%1.3lf]_BETA[%1.3lf]_u0[%s]_ITERS[%d].h5", N, k0, a, b, u0, m_iter * m_end);
	strcat(output_file_name, output_file_data);
	
	// Print file name to screen
	printf("\nOutput File: %s \n\n", output_file_name);


	// open output file and create hyperslabbed datasets 
	open_output_create_slabbed_datasets_lce(&HDF_file_handle, output_file_name, HDF_file_space, HDF_data_set, HDF_mem_space, tot_tsteps, m_end, num_osc, k_range, k1_range, kmin);


	// Create arrays for time and phase order to save after algorithm is finished
	double* time_array      = (double* )malloc(sizeof(double) * (tot_tsteps + 1));
	double* phase_order_R   = (double* )malloc(sizeof(double) * (tot_tsteps + 1));
	double* phase_order_Phi = (double* )malloc(sizeof(double) * (tot_tsteps + 1));

	// ------------------------------
	//  Write Initial Conditions to File
	// ------------------------------
	write_hyperslab_data_d(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], phi, "phi", num_osc, 0);


	// compute triads for initial conditions
	triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
	
	// then write the current modes to this hyperslab
	write_hyperslab_data_d(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], triads, "triads", k_range * k1_range, 0);
	
	phase_order_R[0]   = cabs(triad_phase_order);
	phase_order_Phi[0] = carg(triad_phase_order);
	
	time_array[0] = t0;


	// ------------------------------
	//  Begin Algorithm
	// ------------------------------
	double t = 0.0;
	int iter = 1;
	int save_data_indx = 1;
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

			// print_array_2d_d(RK1_pert, "RK1_pert", num_osc - kmin, num_osc - kmin);

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
			if (iter % save_step == 0) {
				// Write phases
				write_hyperslab_data_d(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], phi, "phi", num_osc, save_data_indx);

				// compute triads for initial conditions
				triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
				
				// write triads
				write_hyperslab_data_d(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], triads, "triads", k_range * k1_range, save_data_indx);

				// save time and phase order parameter
				time_array[save_data_indx]      = iter * dt;
				phase_order_R[save_data_indx]   = cabs(triad_phase_order);
				phase_order_Phi[save_data_indx] = carg(triad_phase_order);

				
				// increment indx for next iteration
				save_data_indx += 1;
			}
			


			// increment
			t    = iter*dt;			
			iter += 1;			
		}

		
		// ------------------------------
		//  Orthonormalize 
		// ------------------------------
		orthonormalize(pert, znorm, num_osc, kmin);


		// ------------------------------
		//  Compute LCEs & Write To File
		// ------------------------------
		for (int i = 0; i < num_osc - kmin; ++i) {
			run_sum[i] = run_sum[i] + log(znorm[i]);
			lce[i]     = run_sum[i] / (t - t0);
		}

		// then write the current LCEs to this hyperslab
		write_hyperslab_data_d(HDF_file_space[2], HDF_data_set[2], HDF_mem_space[2], lce, "lce", num_osc - kmin, m - 1);
		

		// Print update to screen
		if (m % print_every == 0) {
			printf("Iter: %d / %d | t: %5.6lf tsteps: %d | k0:%d alpha: %5.6lf beta: %5.6lf\n", m, m_end, t, m_end * m_iter, k0, a, b);
			printf("k: \n");
			for (int j = 0; j < num_osc - kmin; ++j) {
				printf("%5.6lf ", lce[j]);
			}
			printf("\n\n");
		}


		// ------------------------------
		//  Update For Next Iteration
		// ------------------------------
		T = T + m_iter * dt;
		m += 1;
	}
	// ------------------------------
	//  Write 1D Arrays Using HDF5Lite
	// ------------------------------
	hid_t D2 = 2;
	hid_t D2dims[D2];

	// Write amplitudes
	D2dims[0] = 1;
	D2dims[1] = num_osc;
	H5LTmake_dataset(HDF_file_handle, "Amps", D2, D2dims, H5T_NATIVE_DOUBLE, amp);
	
	// Wtie time
	D2dims[0] = tot_tsteps + 1;
	D2dims[1] = 1;
	H5LTmake_dataset(HDF_file_handle, "Time", D2, D2dims, H5T_NATIVE_DOUBLE, time_array);
	
	// Write Phase Order R
	D2dims[0] = tot_tsteps + 1;
	D2dims[1] = 1;
	H5LTmake_dataset(HDF_file_handle, "PhaseOrderR", D2, D2dims, H5T_NATIVE_DOUBLE, phase_order_R);

	// Write Phase Order Phi
	D2dims[0] = tot_tsteps + 1;
	D2dims[1] = 1;
	H5LTmake_dataset(HDF_file_handle, "PhaseOrderPhi", D2, D2dims, H5T_NATIVE_DOUBLE, phase_order_Phi);



	// ------------------------------
	//  Clean Up & Exit
	// ------------------------------
	// Destroy DFT plans
	fftw_destroy_plan(fftw_plan_r2c);
	fftw_destroy_plan(fftw_plan_c2r);
	
	// Free memory
	free(kx);
	free(amp);
	free(phi);
	free(u_pad);
	free(znorm);
	free(lce);
	free(run_sum);
	free(triads);
	free(phase_order_Phi);
	free(phase_order_R);
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
	H5Sclose( HDF_mem_space[0] );
	H5Dclose( HDF_data_set[0] );
	H5Sclose( HDF_file_space[0] );
	H5Sclose( HDF_mem_space[1] );
	H5Dclose( HDF_data_set[1] );
	H5Sclose( HDF_file_space[1] );
	H5Sclose( HDF_mem_space[2] );
	H5Dclose( HDF_data_set[2] );
	H5Sclose( HDF_file_space[2] );
	H5Fclose(HDF_file_handle);
}
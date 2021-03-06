 // Enda Carroll
// May 2020
// File including functions to perform the Benettin et al., algorithm
// for computing the Lyapunov spectrum and the Ginelli et al. algorithm
// for computing the corresponding Lyapunov vectors of the Phase Only 
// Burgers Equation


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
#include <sys/types.h>
#include <sys/stat.h>


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"





// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
void initial_conditions_lce(double* pert, double* phi, double* amp, fftw_complex* u_z, int* kx, int num_osc, int k0, int kmin, double a, double b, char* IC, int numLEs) {

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
		if (strcmp(IC, "ALIGNED") == 0) {
			if(i <= k0) {
				amp[i] = 0.0;
				phi[i] = 0.0;
				u_z[i] = 0.0 + 0.0 * I;
			} else {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
				phi[i] = (M_PI / 2.0) * (1.0 + 1e-10 * pow((double) i, 0.9));		
				u_z[i] = amp[i] * exp(I * phi[i]);
			}
		} 
		else if (strcmp(IC, "NEW") == 0) {
			if(i <= k0) {
				amp[i] = 0.0;
				phi[i] = 0.0;
				u_z[i] = 0.0 + 0.0 * I;
			} 
			else if (i % 3 == 0){
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
				phi[i] = (M_PI / 2.0) * (1.0 + 1e-10 * pow((double) i, 0.9));	
				u_z[i] = amp[i] * exp(I * phi[i]);
			} 
			else if (i % 3 == 1) {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
				phi[i] = (M_PI / 6.0) * (1.0 + 1e-10 * pow((double) i, 0.9));	
				u_z[i] = amp[i] * exp(I * phi[i]);
			}
			else if (i % 3 == 2) {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
				phi[i] = (5.0 * M_PI / 6.0) * (1.0 + 1e-10 * pow((double) i, 0.9));	
				u_z[i] = amp[i] * exp(I * phi[i]);
			}
		}
		else if (strcmp(IC, "RANDOM") == 0) {
			if(i <= k0) {
				amp[i] = 0.0;
				phi[i] = 0.0;
				u_z[i] = 0.0 + 0.0 * I;
			} else {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );	
				phi[i] = M_PI * ( (double) rand() / (double) RAND_MAX);	
				u_z[i] = amp[i] * exp(I * phi[i]);
			}
		}
		else if (strcmp(IC, "ZERO") == 0) {
			if(i <= k0) {
				amp[i] = 0.0;
				phi[i] = 0.0;
				u_z[i] = 0.0 + 0.0 * I;
			} else {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );	
				phi[i] = 0.0;	
				u_z[i] = amp[i] * exp(I * phi[i]);
			}
		}
		else if (strcmp(IC, "TEST") == 0) {
			if(i <= k0) {
				amp[i] = 0.0;
				phi[i] = 0.0;
				u_z[i] = 0.0 + 0.0 * I;
			} else {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );	
				phi[i] = M_PI / 4.0;	
				u_z[i] = amp[i] * exp(I * phi[i]);
			}
		}

		// fill the perturbation array
		if (i < num_osc - kmin) {
			tmp = i * numLEs;
			for (int j = 0; j < numLEs; ++j)
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

void mem_chk (void *arr_ptr, char *name) {
  if (arr_ptr == NULL ) {
    fprintf(stderr, "ERROR!! |in file \"%s\"-line:%d | Unable to malloc required memory for [%s], now exiting!\n", __FILE__, __LINE__, name);
    exit(1);
  }
}

void conv_direct(fftw_complex* convo, fftw_complex* u_z, int num_osc, int k0) {
	
	// Set the 0 to k0 modes to 0;
	for (int i = 0; i <= k0; ++i) {
		convo[0] = 0.0 + 0.0*I;
	}
	
	// Compute the convolution on the remaining wavenumbers
	int k1;
	for (int kk = k0 + 1; kk < num_osc; ++kk)	{
		for (int k_1 = 1 + kk; k_1 < 2*num_osc; ++k_1)	{
			// Get correct k1 value
			if(k_1 < num_osc) {
				k1 = -num_osc + k_1;
			} else {
				k1 = k_1 - num_osc;
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


void conv_2N_pad(fftw_complex* convo, fftw_complex* uz, fftw_plan *fftw_plan_r2c_ptr, fftw_plan *fftw_plan_c2r_ptr, int n, int num_osc, int k0) {

	// Padded resolution
	int m = 2*n;

	// Normalization factor
	double norm_fact = 1.0 / (double) m;

	// Allocate temporary arrays
	double* u_tmp = (double* )malloc(m*sizeof(double));
	mem_chk(u_tmp, "u_tmp");
	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc((2*num_osc - 1)*sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");

	// write input data to padded array
	for (int i = 0; i < (2*num_osc - 1); ++i)	{
		if(i < num_osc){
			u_z_tmp[i] = uz[i];
		} else {
			u_z_tmp[i] = 0.0 + 0.0*I;
		}
	}

	// // transform back to Real Space
	fftw_execute_dft_c2r((*fftw_plan_c2r_ptr), u_z_tmp, u_tmp);

	// // square
	for (int i = 0; i < m; ++i)	{
		u_tmp[i] = pow(u_tmp[i], 2);
	}

	// here we move the derivative from real to spectral space so that the derivatives can be returned in spectral space
	fftw_execute_dft_r2c((*fftw_plan_r2c_ptr), u_tmp, u_z_tmp);

	// // normalize
	for (int i = 0; i < num_osc; ++i)	{
		if (i <= k0) {
			convo[i] = 0.0 + 0.0*I;
		} else {
			convo[i] = u_z_tmp[i]*(norm_fact);
		}		
	}

	///---------------
	/// Free temp memory
	///---------------
	free(u_tmp);
	fftw_free(u_z_tmp);
}


void po_rhs(double* rhs, fftw_complex* u_z, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0) {

	// Initialize variables
	int m = 2 * n;
	double norm_fac = 1.0 / (double) m;

	fftw_complex pre_fac;

	// Allocate temporary arrays
	double* u_tmp = (double* ) malloc(m*sizeof(double));
	mem_chk(u_tmp, "u_tmp");

	fftw_complex* u_z_tmp = (fftw_complex* ) fftw_malloc((2 * num_osc - 1) * sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");

	///---------------
	/// Convolution
	///---------------
	// Write data to padded array
	for (int i = 0; i < 2*num_osc - 1; ++i)	{
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
			// pre_fac = (-I * kx[k]) / (u_z[k]);
			// rhs[k]  = cimag( pre_fac* (u_z_tmp[k] * norm_fac) );
			pre_fac = -kx[k] / cabs(u_z[k]);
			rhs[k] = pre_fac * creal(cexp(-I * carg(u_z[k]))) * (u_z_tmp[k] * norm_fac);
		}		
	}

	///---------------
	/// Free temp memory
	///---------------
	free(u_tmp);
	fftw_free(u_z_tmp);
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
	double* tmp_rhs = (double* ) fftw_malloc(num_osc*sizeof(double));
	mem_chk(tmp_rhs, "tmp_rhs");

	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");

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
	double* tmp_rhs = (double* ) fftw_malloc(num_osc*sizeof(double));
	mem_chk(tmp_rhs, "tmp_rhs");

	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");

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

	// Frre memory
	free(tmp_rhs);
	fftw_free(u_z_tmp);

	return trans_iters;
}

hid_t create_complex_datatype() {

	// Declare HDF5 datatype variable
	hid_t dtype;

	// error handling var
	herr_t status;
	
	// Create compound datatype for complex numbers
	typedef struct complex_type {
		double re;   // real part 
		double im;   // imaginary part 
	} complex_type;

	struct complex_type cmplex;
	cmplex.re = 0.0;
	cmplex.im = 0.0;

	// create complex compound datatype
	dtype = H5Tcreate(H5T_COMPOUND, sizeof(cmplex));
  	status = H5Tinsert(dtype, "r", offsetof(complex_type,re), H5T_NATIVE_DOUBLE);
  	status = H5Tinsert(dtype, "i", offsetof(complex_type,im), H5T_NATIVE_DOUBLE);

  	return dtype;
}


void get_output_file_name(char* output_file_name, int N, int k0, double a, double b, char* u0, int ntsteps, int m_end, int m_iter, int trans_iters, int numLEs) {

	// Create Output File Locatoin
	char output_dir[512] = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Test/RESULTS";
	// char output_dir[512] = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS/RESULTS";
	char output_dir_tmp[512];
	sprintf(output_dir_tmp,  "_N[%d]_k0[%d]_ALPHA[%1.3lf]_BETA[%1.3lf]_u0[%s]", N, k0, a, b, u0);
	strcat(output_dir, output_dir_tmp);

	// Check if output directory exists, if not make directory
	struct stat st = {0};
	if (stat(output_dir, &st) == -1) {
		mkdir(output_dir, 0700);	  
	}

	// form the filename of the output file	
	char output_file_data[128];
	#ifdef __CLVs
	sprintf(output_file_data, "/CLVData_ITERS[%d,%d,%d]_TRANS[%d]_LEs[%d].h5", ntsteps, m_end, m_iter, trans_iters, numLEs);
	#else
	sprintf(output_file_data, "/LCEData_ITERS[%d,%d,%d]_TRANS[%d]_LEs[%d].h5", ntsteps, m_end, m_iter, trans_iters, numLEs);
	#endif
	strcpy(output_file_name, output_dir);
	strcat(output_file_name, output_file_data);
	
	#ifdef __TRANSIENTS	
	// Print file name to screen
	printf("\nOutput File: %s \n\n", output_file_name);
	printf("\n\tPerforming transient iterations...\n\n");
	#else
	// Print file name to screen
	printf("\nOutput File: %s \n\n", output_file_name);
	#endif	
}



void create_hdf5_slabbed_dset(hid_t* file_handle, char* dset_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, hid_t dtype, hid_t* dset_dims, hid_t* dset_max_dims, hid_t* dset_chunk_dims, const int num_dims) {

	// Error handling variable
	herr_t status;

	// Create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	*file_space = H5Screate_simple(num_dims, dset_dims, dset_max_dims);

	// Must create a propertly list to enable data chunking due to max dimension being unlimited
	// Create property list 
	hid_t plist;
	plist = H5Pcreate(H5P_DATASET_CREATE);

	// Using this property list set the chuncking - stores the chunking info in plist
	status = H5Pset_chunk(plist, num_dims, dset_chunk_dims);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	*data_set = H5Dcreate(*file_handle, dset_name, dtype, *file_space, H5P_DEFAULT, plist, H5P_DEFAULT);
	
	// Create the memory space for the slab
	// setting the max dims to NULL defaults to same size as dims
	*mem_space = H5Screate_simple(num_dims, dset_chunk_dims, NULL);

	// Close the property list object
	status = H5Pclose(plist);
}


void open_output_create_slabbed_datasets_lce(hid_t* file_handle, char* output_file_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, hid_t dtype, int num_t_steps, int num_m_steps, int num_clv_steps, int num_osc, int k_range, int k1_range, int kmin, int numLEs) {

	// ------------------------------
	//  Create file
	// ------------------------------
	// create datafile - H5F_ACC_TRUNC overwrites file if it exists already
	*file_handle = H5Fcreate(output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	// Initialize error handling variable
	herr_t status;

	// create hdf5 dimension arrays for creating the hyperslabs
	static const int dimensions = 2;
	hsize_t dims[dimensions];      // array to hold dims of full evolution data
	hsize_t maxdims[dimensions];   // array to hold max dims of full evolution data
	hsize_t chunkdims[dimensions]; // array to hold dims of the hyperslab chunks


	// ---------------------------------------
	//  Create datasets with hyperslabing
	// ---------------------------------------
	//-----------------------------//
	//---------- PHASES -----------//
	//-----------------------------//
	#ifdef __PHASES
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;             // number of timesteps
	dims[1]      = num_osc;                 // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = num_osc;                 // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = num_osc;                 // 1D chunk of size number of modes

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[0] = H5Screate_simple(dimensions, dims, maxdims);

	// Create the phases dataset
	create_hdf5_slabbed_dset(file_handle, "Phases", &file_space[0], &data_set[0], &mem_space[0], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	#endif

	//-----------------------------//
	//---------- TRIADS -----------//
	//-----------------------------//
	#ifdef __TRIADS
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;             // number of timesteps + initial condition
	dims[1]      = k_range * k1_range;      // size of triads array
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = k_range * k1_range;      // size of triads array
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = k_range*k1_range;        // size of triad array
	
	// Create the dataset for the Triads
	create_hdf5_slabbed_dset(file_handle, "Triads", &file_space[1], &data_set[1], &mem_space[1], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	
	// Create attribute data for the triad dimensions
	hid_t triads_attr, triads_attr_space;

	hsize_t adims[2];
	adims[0] = 1;
	adims[1] = 2;

	triads_attr_space = H5Screate_simple(2, adims, NULL);

	triads_attr = H5Acreate(data_set[1], "Triad_Dims", H5T_NATIVE_INT, triads_attr_space, H5P_DEFAULT, H5P_DEFAULT);

	int triads_dims[2];
	triads_dims[0] = k_range;
	triads_dims[1] = k1_range;

    status = H5Awrite(triads_attr, H5T_NATIVE_INT, triads_dims);

	// close the created attributes obkects
	status = H5Aclose(triads_attr);
    status = H5Sclose(triads_attr_space);	
	#endif

  //-----------------------------------//
	//---------- LCE SPECTRUM -----------//
	//-----------------------------------//
	#ifdef __LCE_ALL
	// initialize the hyperslab arrays
	dims[0]      = num_m_steps;     // number of timesteps
	dims[1]      = numLEs;          // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;   // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = numLEs;          // same as before = number of modes
	chunkdims[0] = 1;               // 1D chunk to be saved 
	chunkdims[1] = numLEs;          // 1D chunk of size number of modes

	// Create the dataset for the LCE spectrum
	create_hdf5_slabbed_dset(file_handle, "LCE", &file_space[2], &data_set[2], &mem_space[2], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	#endif

  //-----------------------------------//
	//----------- R DIAGONAL ------------//
	//-----------------------------------//
	#ifdef __RNORM
	// initialize the hyperslab arrays
	dims[0]      = num_m_steps;     // number of timesteps
	dims[1]      = numLEs;          // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;   // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = numLEs;          // same as before = number of modes
	chunkdims[0] = 1;               // 1D chunk to be saved 
	chunkdims[1] = numLEs;          // 1D chunk of size number of modes

	// Create the dataset for the diagonal of the R matrix
	create_hdf5_slabbed_dset(file_handle, "RNorm", &file_space[3], &data_set[3], &mem_space[3], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	#endif	

	//-----------------------------//
	//----------- CLVs ------------//
	//-----------------------------//
	#ifdef __CLVs
	// initialize the hyperslab arrays
	dims[0]      = num_clv_steps;               // number of timesteps + initial condition
	dims[1]      = (num_osc - kmin) * numLEs;   // size of CLV array
	maxdims[0]   = H5S_UNLIMITED;               // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = (num_osc - kmin) * numLEs;   // size of CLV array
	chunkdims[0] = 1;                           // 1D chunk to be saved 
	chunkdims[1] = (num_osc - kmin) * numLEs;   // size of CLV array

	// Create the dataset for the CLVs
	create_hdf5_slabbed_dset(file_handle, "CLVs", &file_space[4], &data_set[4], &mem_space[4], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);

	// Create attribute data for the CLV dimensions
	hid_t CLV_attr, CLV_attr_space;

	hsize_t CLV_adims[2];
	CLV_adims[0] = 1;
	CLV_adims[1] = 2;

	CLV_attr_space = H5Screate_simple (2, CLV_adims, NULL);

	CLV_attr = H5Acreate(data_set[4], "CLV_Dims", H5T_NATIVE_INT, CLV_attr_space, H5P_DEFAULT, H5P_DEFAULT);

	int CLV_dims[2];
	CLV_dims[0] = (num_osc - kmin);
	CLV_dims[1] = numLEs;

    status = H5Awrite(CLV_attr, H5T_NATIVE_INT, CLV_dims);

	// close the created property list
	status = H5Aclose(CLV_attr);
    status = H5Sclose(CLV_attr_space);
	
  //-------------------------------//
  //----------- ANGLES ------------//
  //-------------------------------//
	#ifdef __ANGLES
	// initialize the hyperslab arrays
	dims[0]      = num_clv_steps;               // number of timesteps + initial condition
	dims[1]      = (num_osc - kmin) * numLEs;   // size of angles array
	maxdims[0]   = H5S_UNLIMITED;               // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = (num_osc - kmin) * numLEs;   // size of angles array
	chunkdims[0] = 1;                           // 1D chunk to be saved 
	chunkdims[1] = (num_osc - kmin) * numLEs;   // size of angles array

	// Create the dataset for the angles between CLVs
	create_hdf5_slabbed_dset(file_handle, "Angles", &file_space[5], &data_set[5], &mem_space[5], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);

	// Create attribute data for the Angles dimensions
	CLV_adims[0] = 1;
	CLV_adims[1] = 2;

	// Create attribute data for the CLV dimensions
	hid_t Angles_attr, Angles_attr_space;

	Angles_attr_space = H5Screate_simple (2, CLV_adims, NULL);

	Angles_attr = H5Acreate(data_set[5], "Angle_Dims", H5T_NATIVE_INT, Angles_attr_space, H5P_DEFAULT, H5P_DEFAULT);

	CLV_dims[0] = (num_osc - kmin);
	CLV_dims[1] = numLEs;

  	status = H5Awrite(Angles_attr, H5T_NATIVE_INT, CLV_dims);

	// close the created property list
	status = H5Aclose(Angles_attr);
    status = H5Sclose(Angles_attr_space);
	#endif
	#endif

  //------------------------------------//
  //----------- MAXIMAL LCE ------------//
  //------------------------------------//
	#ifdef __CLV_MAX
	// initialize the hyperslab arrays
	dims[0]      = num_m_steps;      // number of timesteps
	dims[1]      = (num_osc - kmin); // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;    // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = (num_osc - kmin); // same as before = number of modes
	chunkdims[0] = 1;                // 1D chunk to be saved 
	chunkdims[1] = (num_osc - kmin); // 1D chunk of size number of modes

	// Create the dataset for the CLVs
	create_hdf5_slabbed_dset(file_handle, "MaxCLV", &file_space[6], &data_set[6], &mem_space[6], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	#endif

	//----------------------------//
	//---------- MODES -----------//
	//----------------------------//
	#ifdef __MODES
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;     // number of timesteps
	dims[1]      = num_osc;         // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;   // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = num_osc;         // same as before = number of modes
	chunkdims[0] = 1;               // 1D chunk to be saved 
	chunkdims[1] = num_osc;         // 1D chunk of size number of modes

	// Create the dataset for the Fourier space modes
	create_hdf5_slabbed_dset(file_handle, "Modes", &file_space[7], &data_set[7], &mem_space[7], dtype, dims, maxdims, chunkdims, dimensions);
	#endif

	//---------------------------------//
	//---------- REAL SPACE -----------//
	//---------------------------------//
	#ifdef __REALSPACE
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;     		     // number of timesteps
	dims[1]      = (int) 2 * (num_osc - 1);  // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;   	       // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = (int) 2 * (num_osc - 1);  // same as before = number of modes
	chunkdims[0] = 1;                        // 1D chunk to be saved 
	chunkdims[1] = (int) 2 * (num_osc - 1);  // 1D chunk of size number of modes

	// Create the dataset for the real space solution
	create_hdf5_slabbed_dset(file_handle, "RealSpace", &file_space[8], &data_set[8], &mem_space[8], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	#endif

	//-------------------------------//
	//---------- GRADIENT -----------//
	//-------------------------------//
	#ifdef __GRAD
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;             // number of timesteps
	dims[1]      = (int) 2 * (num_osc - 1); // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;  			    // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = (int) 2 * (num_osc - 1); // same as before = number of modes
	chunkdims[0] = 1;               		    // 1D chunk to be saved 
	chunkdims[1] = (int) 2 * (num_osc - 1); // 1D chunk of size number of modes

	// Create the dataset for the gradient of the real space solution
	create_hdf5_slabbed_dset(file_handle, "Gradient", &file_space[9], &data_set[9], &mem_space[9], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	#endif
}


void write_hyperslab_data(hid_t file_space, hid_t data_set, hid_t mem_space, hid_t dtype, double* data, char* data_name, int n, int index) {

	// Create dimension arrays for hyperslab
	hsize_t start_index[2]; // stores the index in the hyperslabbed dataset to start writing to
	hsize_t count[2];       // stores the size of hyperslab to write to the dataset

	count[0]       = 1;		// 1D slab so first dim is 1
	count[1]       = n;		// 1D slab of size of data array
	start_index[0] = index;	// set the starting row index to index in the global dataset to write slab to
	start_index[1] = 0;		// set column index to 0 to start writing from the first column

	// select appropriate hyperslab 
	if ((H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start_index, NULL, count, NULL)) < 0) {
		fprintf(stderr,"\n!!Error Selecting Hyperslab!! - For [%s] at Index: %d \n", data_name, index);
	}

	// then write the current modes to this hyperslab
	if ((H5Dwrite(data_set, dtype, mem_space, file_space, H5P_DEFAULT, data)) < 0) {
		fprintf(stderr,"\n!!Error Writing Slabbed Data!! - For [%s] at Index: %d \n", data_name, index);
	}
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------

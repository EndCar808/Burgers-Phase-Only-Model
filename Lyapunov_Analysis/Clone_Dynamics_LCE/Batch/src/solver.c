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
void batch_initial_condition(double* phi, double* amp, fftw_complex* u_z, int* kx, int num_osc, int batch, int k0, double a, double b, char* IC, double pert) {

	// set the seed for the random number generator
	srand(123456789);

	int indx;

	// Spectrum cutoff
	double cutoff = ((double) num_osc - 1.0) / 2.0;

	for (int i = 0; i < num_osc; ++i) {

		kx[i] = i;

		// fill the amplitude array
		if (i <= k0) {
			amp[i] = 0.0;
		}
		else {
			amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
		}
		for (int j = 0; j < batch; ++j) {
			indx = i * batch + j;

			// fill amp and phi arrays
			if (strcmp(IC, "ALIGNED") == 0) {
				if(i <= k0) {					
					phi[indx] = 0.0;
					u_z[indx] = 0.0 + 0.0 * I;
				} 
				else {					
					phi[indx] = (M_PI / 2.0) * (1.0 + 1e-10 * pow((double) i, 0.9));		
					u_z[indx] = amp[i] * cexp(I * phi[indx]);
				}
			} 
			else if (strcmp(IC, "NEW") == 0) {
				if(i <= k0) {
					phi[indx] = 0.0;
					u_z[indx] = 0.0 + 0.0 * I;
				} 
				else if (i % 3 == 0){
					phi[indx] = (M_PI / 2.0) * (1.0 + 1e-10 * pow((double) i, 0.9));	
					u_z[indx] = amp[i] * cexp(I * phi[indx]);
				} 
				else if (i % 3 == 1) {
					phi[indx] = (M_PI / 6.0) * (1.0 + 1e-10 * pow((double) i, 0.9));	
					u_z[indx] = amp[i] * cexp(I * phi[indx]);
				}
				else if (i % 3 == 2) {
					phi[indx] = (5.0 * M_PI / 6.0) * (1.0 + 1e-10 * pow((double) i, 0.9));	
					u_z[indx] = amp[i] * cexp(I * phi[indx]);
				}
			}
			else if (strcmp(IC, "RANDOM") == 0) {
				if(i <= k0) {
					phi[indx] = 0.0;
					u_z[indx] = 0.0 + 0.0 * I;
				} else {
					phi[indx] = M_PI * ( (double) rand() / (double) RAND_MAX);	
					u_z[indx] = amp[i] * cexp(I * phi[indx]);
				}
			}
			else if (strcmp(IC, "ZERO") == 0) {
				if(i <= k0) {
					phi[indx] = 0.0;
					u_z[indx] = 0.0 + 0.0 * I;
				} else {
					phi[indx] = 0.0;	
					u_z[indx] = amp[i] * cexp(I * phi[indx]);
				}
			}
			else if (strcmp(IC, "TEST") == 0) {
				if(i <= k0) {
					phi[indx] = 0.0;
					u_z[indx] = 0.0 + 0.0 * I;
				} else {
					phi[indx] = M_PI / 4.0;	
					u_z[indx] = amp[i] * cexp(I * phi[indx]);
				}
			}
		}		
	}

	// Set the initial perturbation
	int col = 1; 
	for (int i = 0; i <= batch; ++i) {
		if (i > k0 && col <= batch) {
			phi[i * batch + col] += pert; 
			col++;
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

void get_output_file_name(char* output_file_name, int N, int k0, double a, double b, char* u0, int ntsteps, int trans_iters) {

	// Create Output File Locatoin
	char output_dir[512] = "../Data/RESULTS/RESULTS";
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
	sprintf(output_file_data,  "/SolverData_ITERS[%d]_TRANS[%d].h5", ntsteps, trans_iters);
	strcpy(output_file_name, output_dir);
	strcat(output_file_name, output_file_data);
	
	#ifdef __STATS	
	// Print file name to screen
	printf("\nOutput File: %s \n\n", output_file_name);
	printf("\n|------------- STATS RUN -------------|\n  Performing transient iterations, no data record...\n\n");
	#else
	// Print file name to screen
	printf("\nOutput File: %s \n\n", output_file_name);
	#endif	
}

void open_output_create_slabbed_datasets(hid_t* file_handle, char* output_file_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, hid_t dtype, int num_t_steps, int num_osc, int k_range, int k1_range) {

	// ------------------------------
	//  Create file
	// ------------------------------
	
	// create datafile - H5F_ACC_TRUNC overwrites file if it exists already
	*file_handle = H5Fcreate(output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);


	// ------------------------------
	//  Create datasets with hyperslabing
	// ------------------------------
	//-----------------------------//
	//---------- PHASES -----------//
	//-----------------------------//
	// create hdf5 dimension arrays for creating the hyperslabs
	static const int dimensions = 2;
	hsize_t dims[dimensions];      // array to hold dims of full evolution data
	hsize_t maxdims[dimensions];   // array to hold max dims of full evolution data
	hsize_t chunkdims[dimensions]; // array to hold dims of the hyperslab chunks

	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;             // number of timesteps
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

	
	//-----------------------------//
	//---------- TRIADS -----------//
	//-----------------------------//
	#ifdef __TRIADS
	// create hdf5 dimension arrays for creating the hyperslabs
	static const int dim2 = 2;
	hsize_t dims2[dim2];      // array to hold dims of full evolution data
	hsize_t maxdims2[dim2];   // array to hold max dims of full evolution data
	hsize_t chunkdims2[dim2]; // array to hold dims of the hyperslab chunks

	// initialize the hyperslab arrays
	dims2[0]      = num_t_steps;             // number of timesteps + initial condition
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
	#endif

	//----------------------------//
	//---------- MODES -----------//
	//----------------------------//
	#ifdef __MODES
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;              // number of timesteps
	dims[1]      = num_osc;                 // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = num_osc;                 // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = num_osc;                 // 1D chunk of size number of modes

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[2] = H5Screate_simple(dimensions, dims, maxdims);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist3;
	plist3 = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist3, dimensions, chunkdims);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[2] = H5Dcreate(*file_handle, "Modes", dtype, file_space[2], H5P_DEFAULT, plist3, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims[0] = 1;
	dims[1] = num_osc;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[2] = H5Screate_simple(dimensions, dims, NULL);

	H5Pclose(plist3);
	#endif

	//--------------------------------//
	//---------- REALSPACE -----------//
	//--------------------------------//
	#ifdef __REALSPACE
	int N = 2 * (num_osc - 1);
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;             // number of timesteps
	dims[1]      = N;                       // number of collocation points
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = N;                       // number of collocation points
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = N;                       // number of collocation points

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[3] = H5Screate_simple(dimensions, dims, maxdims);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist4;
	plist4 = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist4, dimensions, chunkdims);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[3] = H5Dcreate(*file_handle, "RealSpace", H5T_NATIVE_DOUBLE, file_space[3], H5P_DEFAULT, plist4, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims[0] = 1;
	dims[1] = N;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[3] = H5Screate_simple(dimensions, dims, NULL);

	H5Pclose(plist4);
	#endif
}

hid_t create_complex_datatype() {
	hid_t dtype;
	// Create compound datatype for complex numbers
	typedef struct complex_type {
		double re;   // real part 
		double im;   // imaginary part 
	} complex_type;

	struct complex_type cmplex;
	cmplex.re = 0.0;
	cmplex.im = 0.0;

	// create complex compound datatype
	dtype = H5Tcreate (H5T_COMPOUND, sizeof(cmplex));
  	H5Tinsert(dtype, "r", offsetof(complex_type,re), H5T_NATIVE_DOUBLE);
  	H5Tinsert(dtype, "i", offsetof(complex_type,im), H5T_NATIVE_DOUBLE);

  	return dtype;
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
		printf("\n!!Error Selecting Hyperslab!! - For %s at Index: %d \n", data_name, index);
	}

	// then write the current modes to this hyperslab
	if ((H5Dwrite(data_set, dtype, mem_space, file_space, H5P_DEFAULT, data)) < 0) {
		printf("\n!!Error Writing Slabbed Data!! - For %s at Index: %d \n", data_name, index);
	}
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

	double* u_tmp = (double* )malloc(n*sizeof(double));
	mem_chk(u_tmp, "u_tmp");

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
	double* u_tmp = (double* ) malloc( m * sizeof(double));
	mem_chk(u_tmp, "u_tmp");

	fftw_complex* u_z_tmp = (fftw_complex* ) fftw_malloc((2 * num_osc - 1) * sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");

	///---------------
	/// Convolution
	///---------------
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


void po_rhs_batch(double* rhs, fftw_complex* u_z, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0, int batch) {

	// Initialize variables
	int m = 2 * n;
	double norm_fac = 1.0 / (double) m;
	int indx;

	fftw_complex pre_fac;

	// Allocate temporary arrays
	double* u_tmp = (double* )malloc(m * batch * sizeof(double));
	mem_chk(u_tmp, "u_tmp");

	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc((2 * num_osc - 1) * batch * sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");

	///---------------
	/// Convolution
	///---------------
	// Write data to padded array
	for (int i = 0; i < (2 * num_osc - 1); ++i)	{
		for (int j = 0; j < batch; ++j)
		{
			indx = i * batch + j;
			if(i < num_osc){
				u_z_tmp[indx] = u_z[indx];
			} else {
				u_z_tmp[indx] = 0.0 + 0.0*I;
			}
		}
	}

	// transform back to Real Space
	fftw_execute_dft_c2r((*plan_c2r_pad), u_z_tmp, u_tmp);

	// multiplication in real space
	for (int i = 0; i < m; ++i)	{
		for (int j = 0; j < batch; ++j)
		{
			indx = i * batch + j;
			u_tmp[indx] = pow(u_tmp[indx], 2);
		}
	}


	// transform forward to Fourier space
	fftw_execute_dft_r2c((*plan_r2c_pad), u_tmp, u_z_tmp);

	///---------------
	/// RHS
	///---------------
	for (int k = 0; k < num_osc; ++k) {
		for (int j = 0; j < batch; ++j)
		{
			indx = k * batch + j;
			if (k <= k0) {
				rhs[indx] = 0.0;
			} else {
				pre_fac = (-I * kx[k]) / (2.0 * u_z[indx]);
				rhs[indx]  = cimag( pre_fac* (u_z_tmp[indx] * norm_fac) );
			}		
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
			u_z_tmp[i] = amps[i] * cexp(I * 0.0);
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

double get_timestep_batch(double* amps, int* kx, int n, int m, int num_osc, int k0) {

	// Initialize vars
	double dt;
	double max_val;

	// Initialize temp memory
	double* tmp_rhs = (double* ) fftw_malloc(num_osc*sizeof(double));
	mem_chk(tmp_rhs, "tmp_rhs");
	double* u_pad   = (double* ) fftw_malloc(m * sizeof(double));
	mem_chk(u_pad, "u_pad");

	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");
	fftw_complex* u_z_pad = (fftw_complex* )fftw_malloc((m / 2 + 1)*sizeof(fftw_complex));
	mem_chk(u_z_pad, "u_z_pad");

	fftw_plan plan_c2r, plan_r2c;
	plan_r2c = fftw_plan_dft_r2c_1d(m, u_pad, u_z_pad, FFTW_PRESERVE_INPUT); 
	plan_c2r = fftw_plan_dft_c2r_1d(m, u_z_pad, u_pad, FFTW_PRESERVE_INPUT);

	// Create modes for RHS evaluation
	for (int i = 0; i < num_osc; ++i) {
		if (i <= k0) {
			u_z_tmp[i] = 0.0 + 0.0*I;
		} else {
			u_z_tmp[i] = amps[i] * cexp(I * 0.0);
		}
	}

	// Call the RHS
	po_rhs(tmp_rhs, u_z_tmp, &plan_c2r, &plan_r2c, kx, n, num_osc, k0);

	
	// Find  the fastest moving oscillator
	max(tmp_rhs, num_osc, k0, &max_val);


	// Get timestep
	dt = 1.0 / max_val;

	free(tmp_rhs);
	free(u_pad);
	fftw_free(u_z_tmp);
	fftw_free(u_z_pad);
	
	return dt;
}

int get_transient_iters_batch(double* amps, int* kx, int n, int m, int num_osc, int k0) {

	// Initialize vars
	int trans_iters;
	double min_val;
	double max_val;
	double trans_ratio;
	int trans_mag;

	// Initialize temp memory
	double* tmp_rhs = (double* ) fftw_malloc(num_osc*sizeof(double));
	mem_chk(tmp_rhs, "tmp_rhs");
	double* u_pad   = (double* ) fftw_malloc(m * sizeof(double));
	mem_chk(u_pad, "u_pad");

	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");
	fftw_complex* u_z_pad = (fftw_complex* )fftw_malloc((m / 2 + 1)*sizeof(fftw_complex));
	mem_chk(u_z_pad, "u_z_pad");

	fftw_plan plan_c2r, plan_r2c;
	plan_r2c = fftw_plan_dft_r2c_1d(m, u_pad, u_z_pad, FFTW_PRESERVE_INPUT); 
	plan_c2r = fftw_plan_dft_c2r_1d(m, u_z_pad, u_pad, FFTW_PRESERVE_INPUT);


	// Create modes for RHS evaluation
	for (int i = 0; i < num_osc; ++i) {
		if (i <= k0) {
			u_z_tmp[i] = 0.0 + 0.0*I;
		} else {
			u_z_tmp[i] = amps[i] * cexp(I * 0.0);
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

	free(tmp_rhs);
	free(u_pad);
	fftw_free(u_z_tmp);
	fftw_free(u_z_pad);

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


void cloneDynamics(int N, int k0, double a, double b, int batch, int mITERS, int mEND, char* u0, double pert) {

	// ------------------------------
	//  Variable Definitions
	// ------------------------------
	// Number of modes
	int num_osc = (N / 2) + 1; 

	// padded array size
	int M = 2 * N;

	// Forcing wavenumber
	int kmin = k0 + 1;
	int kmax = num_osc - 1;

	// Triad phases arrayzz
	int k_range  = kmax - kmin + 1;
	int k1_range = (int)((kmax - kmin + 1)/ 2.0);
	int dof = num_osc - kmin;

	// print update every x iterations
	int print_every = (mEND >= 10 ) ? (int)((double)mEND * 0.1) : 10;

	int indx;
	int tmp;	
	double lce_sum;
	double dim_sum;
	int dim_indx;
	
	// ------------------------------
	//  Allocate memory
	// ------------------------------
	// wavenumbers
	int* kx = (int* ) malloc(num_osc * sizeof(int));
	mem_chk(kx, "kx");

	// Oscillator arrays
	double* amp = (double* ) malloc(num_osc * sizeof(double));
	mem_chk(amp, "amp");
	double* phi = (double* ) malloc(num_osc * batch * sizeof(double));
	mem_chk(phi, "phi");
	double* lce = (double* ) malloc(dof * sizeof(double));
	mem_chk(lce, "lce");
	double* run_sum = (double* ) malloc(dof * sizeof(double));
	mem_chk(run_sum, "run_sum");
	double* rnorm = (double* ) malloc(dof * sizeof(double));
	mem_chk(rnorm, "rnorm");
	double* pertMat = (double* ) malloc(dof * dof * sizeof(double));
	mem_chk(pertMat, "pertMat");

	// modes array
	fftw_complex* u_z = (fftw_complex* ) fftw_malloc(num_osc * batch * sizeof(fftw_complex));
	mem_chk(u_z, "u_z");
	#ifdef __REALSPACE
	double* u = (double* ) malloc(N * batch * sizeof(double));
	mem_chk(u, "u");
	#endif

	// padded solution arrays
	double* u_pad = (double* ) malloc(M * batch * sizeof(double));
	mem_chk(u_pad, "u_pad");
	fftw_complex* u_z_pad = (fftw_complex* ) fftw_malloc((2 * num_osc - 1) * batch * sizeof(fftw_complex));
	mem_chk(u_z_pad, "u_z_pad");

	#ifdef __TRIADS
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
	fftw_complex triad_phase_order = 0.0 + I * 0.0;
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
	double*	RK1 = (double* )fftw_malloc(num_osc * batch * sizeof(double));
	double* RK2 = (double* )fftw_malloc(num_osc * batch * sizeof(double));
	double* RK3 = (double* )fftw_malloc(num_osc * batch * sizeof(double));
	double* RK4 = (double* )fftw_malloc(num_osc * batch * sizeof(double));
	mem_chk(RK1, "RK1");
	mem_chk(RK2, "RK2");
	mem_chk(RK3, "RK3");
	mem_chk(RK4, "RK4");

	// temporary memory to store stages
	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc(num_osc * batch * sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");
	

	// ------------------------------
	//  Create FFTW plans
	// ------------------------------
	fftw_plan fftw_plan_r2c_pad, fftw_plan_c2r_pad;	

	// Batch plan variables
	int rank = 1;               // Dimension of the transform to perform -> 1D on each of the cols
	int nSize[1]  = {M};          // Array containing the size of the transform in each dim to perform -> size n transform
	int inStride  = batch;		// Distance between two successive input elements  -> Distance between elements in a single transform
	int outStride = batch;      // Distance between two successive output elements -> Distance between elements in a single transform
	int inDist    = 1;			// Distance between input elements -> 1 since it is a simple transform
	int outDist   = 1;           // Distance between output batches -> 1 since it is a simple transform
	int *inEmbed  = nSize;		// Since we are doing a full transform of the cols this should be equal to nSize, if we were doing a transform on a subset or sub array these dim sizes would be less than nSize
	int *outEmbed = nSize;		// Since we are doing a full transform of the cols this should be equal to nSize, if we were doing a transform on a subset or sub array these dim sizes would be less than nSize
	fftw_plan_r2c_pad = fftw_plan_many_dft_r2c(rank, nSize, batch, u_pad, inEmbed, inStride, inDist, u_z_pad, outEmbed, outStride, outDist, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r_pad = fftw_plan_many_dft_c2r(rank, nSize, batch, u_z_pad, outEmbed, outStride, outDist, u_pad, inEmbed, inStride, inDist, FFTW_PRESERVE_INPUT); 
	#ifdef __REALSPACE
	fftw_plan fftw_plan_r2c, fftw_plan_c2r;

	// Batch padded plan variables
    nSize[0] = N;      // Array containing the size of the transform to perform -> N for unpadded arrays
	fftw_plan_r2c = fftw_plan_many_dft_r2c(rank, nSize, batch, u, inEmbed, inStride, inDist, u_z, outEmbed, outStride, outDist, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r = fftw_plan_many_dft_c2r(rank, nSize, batch, u_z, outEmbed, outStride, outDist, u, inEmbed, inStride, inDist, FFTW_PRESERVE_INPUT);
	#endif


	// ------------------------------
	//  Generate Initial Conditions
	// ------------------------------
	batch_initial_condition(phi, amp, u_z, kx, num_osc, batch, k0, a, b, u0, pert);

	// Print IC if small system size
	if (N <= 32) {
		for (int i = 0; i < num_osc; ++i) {
			for (int j = 0; j < batch; ++j)
			{
				printf("phi[%d]: %5.16lf \t", i, phi[i * batch + j]);
			}	
			printf("\n");		
		}
		printf("\n\n");
		for (int i = 0; i < num_osc; ++i) {
			for (int j = 0; j < batch; ++j)
			{
				printf("u_z[%d]:%5.16lf %5.16lfI\t", i, creal(u_z[i * batch + j]), cimag(u_z[i * batch + j]));
			}			
			printf("\n");
		}
		printf("\n\n");
	}
	

	// ------------------------------
	//  Get Timestep & Time variables
	// ------------------------------
	double dt   = get_timestep_batch(amp, kx, N, M, num_osc, k0);

	// If calculating stats or integrating until transient get required iters
	#ifdef __TRANSIENTS
	int trans_iters    = get_transient_iters_batch(amp, kx, N, M, num_osc, k0); 
	#else 
	int trans_iters    = 0; 
	#endif

	// Saving variables
	int tot_m_save_steps = (int) mEND / SAVE_LCE_STEP;
	int tot_t_save_steps = (int) ((mITERS * mEND) / SAVE_DATA_STEP);	

	// LCE algorithm varibales
	#ifdef __TRANSIENTS
	int trans_m = (int) ceil(trans_iters / mITERS);
	#else
	int trans_m = 0;
	#endif	
	int max_mITERS = mEND + trans_m;

	// Time variables	
	double t0 = 0.0;
	double T  = t0 + (trans_iters + mITERS) * dt;

	printf("\n\tTimeStep: %20.16lf\n\n", dt);
	// // ------------------------------
	// //  HDF5 File Create
	// // ------------------------------
	// // Create the HDF5 file handle
	// hid_t HDF_Outputfile_handle;


	// // create hdf5 handle identifiers for hyperslabing the full evolution data
	// hid_t HDF_file_space[4];
	// hid_t HDF_data_set[4];
	// hid_t HDF_mem_space[4];

	// // get output file name
	// char output_file_name[512];
	// get_output_file_name(output_file_name, N, k0, a, b, u0, ntsteps, trans_iters);

	// // Create complex datatype for hdf5 file if modes are being recorded
	// #ifdef __MODES
	// hid_t COMPLEX_DATATYPE = create_complex_datatype();
	// #else
	// hid_t COMPLEX_DATATYPE = -1;
	// #endif

	// // open output file and create hyperslabbed datasets 
	// open_output_create_slabbed_datasets(&HDF_Outputfile_handle, output_file_name, HDF_file_space, HDF_data_set, HDF_mem_space, COMPLEX_DATATYPE, num_save_steps, num_osc, k_range, k1_range);


	

	// // ------------------------------
	// //  Write Initial Conditions to File
	// // ------------------------------
	// // Create non chunked data arrays
	// double* time_array      = (double* )malloc(sizeof(double) * (num_save_steps));
	// mem_chk(time_array, "time_array");
	// #ifdef __TRIADS
	// double* phase_order_R   = (double* )malloc(sizeof(double) * (num_save_steps));
	// double* phase_order_Phi = (double* )malloc(sizeof(double) * (num_save_steps));
	// mem_chk(phase_order_R, "phase_order_R");
	// mem_chk(phase_order_Phi, "phase_order_Phi");	
	// #endif

	// // Write initial condition if transient iterations are not being performed
	// #ifndef __TRANSIENTS
	// // Write Initial condition for phases
	// write_hyperslab_data(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], H5T_NATIVE_DOUBLE, phi, "phi", num_osc, 0);

	// #ifdef __TRIADS
	// // compute triads for initial conditions
	// triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
	
	// // // then write the current modes to this hyperslab
	// write_hyperslab_data(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], H5T_NATIVE_DOUBLE, triads, "triads", k_range * k1_range, 0);

	// // Write initial order param values
	// phase_order_R[0]   = cabs(triad_phase_order);
	// phase_order_Phi[0] = carg(triad_phase_order);
	// #endif __TRIADS

	// #ifdef __MODES
	// // Write Initial condition for modes
	// write_hyperslab_data(HDF_file_space[2], HDF_data_set[2], HDF_mem_space[2], COMPLEX_DATATYPE, u_z, "u_z", num_osc, 0);
	// #endif

	// #ifdef __REALSPACE 
	// // transform back to Real Space
	// fftw_execute_dft_c2r(fftw_plan_c2r, u_z, u);

	// // Write Initial condition for real space
	// write_hyperslab_data(HDF_file_space[3], HDF_data_set[3], HDF_mem_space[3], H5T_NATIVE_DOUBLE, u, "u", N, 0);
	// #endif
	
	// // write initial time
	// time_array[0] = t0;
	// #endif

	
	// ------------------------------
	//  Integration & Looping Params
	// ------------------------------
	int m    = 1;
	int iter = 1;
	double t = 0.0;	
	#ifdef __TRANSIENTS
	int save_data_indx = 0;
	#else
	int save_data_indx = 1;
	#endif


	// ------------------------------
	//  Begin Algorithm
	// ------------------------------
	while(m <= mEND) {
		
		// ------------------------------
		//  Begin Integration
		// ------------------------------
		for (int p = 0; p < mITERS; ++p)
		{

			// Construct the modes
			for (int i = 0; i < num_osc; ++i) {
				for (int j = 0; j < batch; ++j) {
					indx = i * batch + j;
					u_z_tmp[indx] = amp[i] * cexp(I * phi[indx]);			
				}			
			}

			//////////////
			// STAGES
			//////////////
			/*---------- STAGE 1 ----------*/
			// find RHS first and then update stage
			po_rhs_batch(RK1, u_z_tmp, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, kx, N, num_osc, k0, batch);
			for (int i = 0; i < num_osc; ++i) {
				for (int j = 0; j < batch; ++j) {
					indx = i * batch + j;
					u_z_tmp[indx] = amp[i] * cexp(I * (phi[indx] + A21 * dt * RK1[indx]));
				}
			}

			// for (int i = 0; i < num_osc; ++i) {
			// 	for (int j = 0; j < batch; ++j)
			// 	{
			// 		printf("u_z_tmp[%d]:%5.10lf %5.10lfI \t", i, creal(u_z_tmp[i * batch + j]), cimag(u_z_tmp[i * batch + j]));
			// 	}	
			// 	printf("\n");		
			// }
			// printf("\n\n");

			/*---------- STAGE 2 ----------*/
			// find RHS first and then update stage
			po_rhs_batch(RK2, u_z_tmp, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, kx, N, num_osc, k0, batch);
			for (int i = 0; i < num_osc; ++i) {
				for (int j = 0; j < batch; ++j) {
					indx = i * batch + j;
					u_z_tmp[indx] = amp[i] * cexp(I * (phi[indx] + A32 * dt * RK2[indx]));
				}
			}

			/*---------- STAGE 3 ----------*/
			// find RHS first and then update stage
			po_rhs_batch(RK3, u_z_tmp, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, kx, N, num_osc, k0, batch);
			for (int i = 0; i < num_osc; ++i) {
				for (int j = 0; j < batch; ++j) {
					indx = i * batch + j;
					u_z_tmp[indx] = amp[i] * cexp(I * (phi[indx] + A43 * dt * RK3[indx]));
				}
			}


			/*---------- STAGE 4 ----------*/
			// find RHS first and then update 
			po_rhs_batch(RK4, u_z_tmp, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, kx, N, num_osc, k0, batch);
			

			//////////////
			// Update
			//////////////
			for (int i = 0; i < num_osc; ++i) {
				for (int j = 0; j < batch; ++j) {
					indx   = i * batch + j;
					phi[indx] = phi[indx] + (dt * B1) * RK1[indx] + (dt * B2) * RK2[indx] + (dt * B3) * RK3[indx] + (dt * B4) * RK4[indx];  
				}
			}
		
			// for (int i = 0; i < num_osc; ++i) {
			// 	for (int j = 0; j < batch; ++j)
			// 	{
			// 		printf("RK3[%d]: %5.16lf \t", i, RK3[i * batch + j]);
			// 	}	
			// 	printf("\n");		
			// }
			// printf("\n\n");

			// ////////////
			// // Print to file
			// ////////////
			// if ((iter > trans_iters) && (iter % save_step == 0)) {
			// 	// Write phases
			// 	write_hyperslab_data(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], H5T_NATIVE_DOUBLE, phi, "phi", num_osc, save_data_indx);

			// 	#ifdef __TRIADS
			// 	// compute triads for initial conditions
			// 	triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
				
			// 	// write triads
			// 	write_hyperslab_data(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], H5T_NATIVE_DOUBLE, triads, "triads", k_range * k1_range, save_data_indx);

			// 	phase_order_R[save_data_indx]   = cabs(triad_phase_order);
			// 	phase_order_Phi[save_data_indx] = carg(triad_phase_order);
			// 	#endif
			// 	#ifdef __MODES || __REALSPACE
			// 	// Construct the modes
			// 	for (int i = 0; i < num_osc; ++i) {
			// 		u_z[i] = amp[i] * cexp(I * phi[i]);
			// 	}
			// 	#endif
			// 	#ifdef __MODES
			// 	// Write Initial condition for modes
			// 	write_hyperslab_data(HDF_file_space[2], HDF_data_set[2], HDF_mem_space[2], COMPLEX_DATATYPE, u_z, "u_z", num_osc, save_data_indx);
			// 	#endif
			// 	#ifdef __REALSPACE 
			// 	// transform back to Real Space
			// 	fftw_execute_dft_c2r(fftw_plan_c2r, u_z, u);

			// 	// Write Initial condition for real space
			// 	write_hyperslab_data(HDF_file_space[3], HDF_data_set[3], HDF_mem_space[3], H5T_NATIVE_DOUBLE, u, "u", N, save_data_indx);
			// 	#endif


			// 	// save time and phase order parameter
			// 	time_array[save_data_indx]      = iter * dt;

			// 	// increment indx for next iteration
			// 	save_data_indx++;
			// }
			// // Check if transient iterations has been reached
			// #ifdef __TRANSIENTS
			// if (iter == trans_iters) printf("\n|---- TRANSIENT ITERATIONS COMPLETE ----|\n  Iterations Performed:\t%d\n  Iterations Left:\t%d\n\n", trans_iters, iters);
			// #endif

					
			// increment
			t   = iter*dt;
			iter++;
		}
		// ------------------------------
		//  End Integration
		// ------------------------------
		

		// ------------------------------
		//  Orthonormalize
		// ------------------------------
		// Create perturbation matrix
		for (int i = 0; i < num_osc; ++i)
		{
			for (int j = 0; j < batch; ++j)
			{
				if (i > k0 && j >= 1) {
					pertMat[(i - kmin) * dof + (j - 1)] = phi[i * batch] - phi[i * batch + j];
				}
				// printf("phi_tmp[%d]: %5.16lf \t", i, phi_tmp[i * batch + j]);
			}
			// printf("\n");
		}
		// printf("\n\n");

		// reorthonormalize the bases
		orthonormalize(pertMat, rnorm, num_osc, kmin);

		// printf("After:\n");
		// for (int i = 0; i < num_osc - kmin; ++i)
		// {
		// 	for (int j = 0; j < num_osc - kmin; ++j)
		// 	{
		// 		printf("pert[%d]: %5.16lf \t", i*(num_osc - kmin) + j, pertMat[i*(num_osc - kmin) + j]);
		// 	}
		// 	printf("\n");
		// }
		// printf("\n\n");


		// for (int i = 0; i < num_osc; ++i)
		// {
		// 	for (int j = 0; j < num_osc - kmin; ++j)
		// 	{
		// 		if (i <= k0) {
		// 			printf("phi[%d]: %5.16lf \t", i*(num_osc - kmin) + j, 0.0);
		// 		}else {
		// 			printf("phi[%d]: %5.16lf \t", i*(num_osc - kmin) + j, phi[i] + pert * pertMat[(i - kmin)*(num_osc - kmin) + (j)]);
		// 		}
		// 	}
		// 	printf("\n");
		// }
		// printf("\n\n");
		

		// ------------------------------
		//  Compute LCEs & Write To File
		// ------------------------------
		if (m > trans_m) {
			for (int i = 0; i < dof; ++i) {
				// Compute LCE
				run_sum[i] = run_sum[i] + log(rnorm[i]);
				lce[i]     = run_sum[i] / (t - t0);
			}

			// // then write the current LCEs to this hyperslab
			// if (m % SAVE_LCE_STEP == 0) {			
			// 	write_hyperslab_data_d(HDF_file_space[2], HDF_data_set[2], HDF_mem_space[2], lce, "lce", num_osc - kmin, save_lce_indx - 1);

			// 	save_lce_indx += 1;
			// }

			// Print update to screen
			if (m % print_every == 0) {
				lce_sum  = 0.0;
				dim_sum  = 0.0;
				dim_indx = 0;
				for (int i = 0; i < dof; ++i) {
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
				printf("Iter: %d / %d | t: %5.6lf tsteps: %d | k0:%d alpha: %5.6lf beta: %5.6lf | Sum: %5.9lf | Dim: %5.9lf\n", m, mEND, t, mEND * mITERS, k0, a, b, lce_sum, (dim_indx + (dim_sum / fabs(lce[dim_indx]))));
				printf("k: \n");
				for (int j = 0; j < dof; ++j) {
					printf("%5.6lf ", lce[j]);
				}
				printf("\n\n");
			}
		}


		// ------------------------------
		//  Update For Next Iteration
		// ------------------------------
		T = T + mITERS * dt;
		m += 1;
		#ifdef __TRANSIENTS
		if (m - 1 == trans_m) {
			printf("\n\t!!Transient Iterations Complete!! - Iters: %d\n\n", iter - 1);
		}
		#endif

		// update the clone trajectories
		for (int i = 0; i < num_osc; ++i) {
			for (int j = 1; j < batch; ++j) {	
				if (i <= k0) {
					phi[i * batch + j] = 0.0;
				}
				else {
					phi[i * batch + j] = phi[i * batch] + pert * pertMat[(i - kmin) * dof + (j - 1)];
				}
			}
		}
	}	
	// ------------------------------
	//  End Algorithm
	// ------------------------------
	

	// // ------------------------------
	// //  Write 1D Arrays Using HDF5Lite
	// // ------------------------------
	// hid_t D1 = 1;
	// hid_t D1dims[D1];

	// // Write amplitudes
	// D1dims[0] = num_osc;
	// if ( (H5LTmake_dataset(HDF_Outputfile_handle, "Amps", D1, D1dims, H5T_NATIVE_DOUBLE, amp)) < 0) {
	// 	printf("\n\n!!Failed to make - Amps - Dataset!!\n\n");
	// }
	
	// // Wtie time
	// D1dims[0] = num_save_steps;
	// if ( (H5LTmake_dataset(HDF_Outputfile_handle, "Time", D1, D1dims, H5T_NATIVE_DOUBLE, time_array)) < 0) {
	// 	printf("\n\n!!Failed to make - Time - Dataset!!\n\n");
	// }
	
	// #ifdef __TRIADS
	// // Write Phase Order R
	// D1dims[0] = num_save_steps;
	// if ( (H5LTmake_dataset(HDF_Outputfile_handle, "PhaseOrderR", D1, D1dims, H5T_NATIVE_DOUBLE, phase_order_R)) < 0) {
	// 	printf("\n\n!!Failed to make - PhaseOrderR - Dataset!!\n\n");
	// }
	// // Write Phase Order Phi
	// D1dims[0] = num_save_steps;
	// if ( (H5LTmake_dataset(HDF_Outputfile_handle, "PhaseOrderPhi", D1, D1dims, H5T_NATIVE_DOUBLE, phase_order_Phi)) < 0) {
	// 	printf("\n\n!!Failed to make - PhaseOrderPhi - Dataset!!\n\n");
	// }
	// #endif
	

	// for (int i = 0; i < num_osc; ++i) {
	// 	for (int j = 0; j < batch; ++j)
	// 	{
	// 		printf("phi[%d]: %5.16lf \t", i, phi[i * batch + j]);
	// 	}	
	// 	printf("\n");		
	// }
	// printf("\n\n");
	


	// ------------------------------
	//  Clean Up
	// ------------------------------
	// destroy fftw plans
	#ifdef __REALSPACE
	fftw_destroy_plan(fftw_plan_r2c);
	fftw_destroy_plan(fftw_plan_c2r);
	#endif
	fftw_destroy_plan(fftw_plan_r2c_pad);
	fftw_destroy_plan(fftw_plan_c2r_pad);



	// free memory
	free(kx);
	free(u_pad);
	#ifdef __REALSPACE
	free(u);
	#endif
	#ifdef __TRIADS
	free(triads);	
	free(phase_order_Phi);
	free(phase_order_R);
	#endif
	// free(time_array);
	free(rnorm);
	free(lce);
	free(run_sum);
	free(pertMat);
	fftw_free(u_z);
	fftw_free(RK1);
	fftw_free(RK2);
	fftw_free(RK3);
	fftw_free(RK4);
	fftw_free(u_z_tmp);
	fftw_free(u_z_pad);

	// // close HDF5 handles
	// H5Sclose( HDF_mem_space[0] );
	// H5Dclose( HDF_data_set[0] );
	// H5Sclose( HDF_file_space[0] );
	// #ifdef __TRIADS
	// H5Sclose( HDF_mem_space[1] );
	// H5Dclose( HDF_data_set[1] );
	// H5Sclose( HDF_file_space[1] );
	// #endif
	// #ifdef __MODES
	// H5Sclose( HDF_mem_space[2] );
	// H5Dclose( HDF_data_set[2] );
	// H5Sclose( HDF_file_space[2] );
	// #endif 
	// #ifdef __REALSPACE
	// H5Sclose( HDF_mem_space[3] );
	// H5Dclose( HDF_data_set[3] );
	// H5Sclose( HDF_file_space[3] );
	// #endif

	// // Close pipeline to output file
	// H5Fclose(HDF_Outputfile_handle);
}

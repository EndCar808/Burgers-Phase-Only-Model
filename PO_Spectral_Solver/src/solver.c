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
void initial_condition(double* phi, double* amp, fftw_complex* u_z, int* kx, int num_osc, int k0, double a, double b, char* IC) {

	// set the seed for the random number generator
	srand(123456789);

	// Spectrum cutoff
	double cutoff = ((double) num_osc - 1.0) / 2.0;

	for (int i = 0; i < num_osc; ++i) {

		// fill the wavenumbers array
		kx[i] = i;

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
	u_tmp = (double* ) malloc( m * sizeof(double));

	fftw_complex* u_z_tmp;
	u_z_tmp = (fftw_complex* ) fftw_malloc((2 * num_osc - 1) * sizeof(fftw_complex));

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


void solver(int N, int k0, double a, double b, int iters, int save_step, char* u0) {

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

	// print update every x iterations
	int print_update = (iters >= 10 ) ? (int)((double)iters * 0.1) : 1;

	int indx;
	int tmp;	
	
	// ------------------------------
	//  Allocate memory
	// ------------------------------
	// wavenumbers
	int* kx = (int* ) malloc(num_osc * sizeof(int));

	// Oscillator arrays
	double* amp = (double* ) malloc(num_osc * sizeof(double));
	double* phi = (double* ) malloc(num_osc * sizeof(double));

	// modes array
	fftw_complex* u_z = (fftw_complex* ) fftw_malloc(num_osc * sizeof(fftw_complex));
	#ifdef __REALSPACE
	double* u = (double* ) malloc(N * sizeof(double));
	#endif

	// padded solution arrays
	double* u_pad = (double* ) malloc(M * sizeof(double));
	fftw_complex* u_z_pad = (fftw_complex* ) fftw_malloc((2 * num_osc - 1) * sizeof(fftw_complex));

	#ifdef __TRIADS
	// Triad phases array
	int k_range  = kmax - kmin + 1;
	int k1_range = (int)((kmax - kmin + 1)/ 2.0);

	double* triads = (double* )malloc(k_range * k1_range * sizeof(double));
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
	double*	RK1 = (double* )fftw_malloc(num_osc*sizeof(double));
	double* RK2 = (double* )fftw_malloc(num_osc*sizeof(double));
	double* RK3 = (double* )fftw_malloc(num_osc*sizeof(double));
	double* RK4 = (double* )fftw_malloc(num_osc*sizeof(double));

	// temporary memory to store stages
	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));
	

	// ------------------------------
	//  Create FFTW plans
	// ------------------------------
	// create fftw3 plans objects
	fftw_plan fftw_plan_r2c_pad, fftw_plan_c2r_pad;
	
	// create plans - ensure no overwriting - fill arrays after
	fftw_plan_r2c_pad = fftw_plan_dft_r2c_1d(M, u_pad, u_z_pad, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r_pad = fftw_plan_dft_c2r_1d(M, u_z_pad, u_pad, FFTW_PRESERVE_INPUT);
	#ifdef __REALSPACE
	fftw_plan fftw_plan_r2c, fftw_plan_c2r;
	fftw_plan_r2c = fftw_plan_dft_r2c_1d(N, u, u_z, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(N, u_z, u, FFTW_PRESERVE_INPUT);
	#endif


	// ------------------------------
	//  Generate Initial Conditions
	// ------------------------------
	initial_condition(phi, amp, u_z, kx, num_osc, k0, a, b, u0);
	
	// Print IC if small system size
	if (N <= 32) {
		for (int i = 0; i < num_osc; ++i) {
			printf("k[%d]: %d | a: %5.15lf   p: %5.16lf   | u_z[%d]: %5.16lf  %5.16lfI \n", i, kx[i], amp[i], phi[i], i, creal(u_z[i]), cimag(u_z[i]));
		}
		printf("\n");
	}
	

	// ------------------------------
	//  Get Timestep & Time variables
	// ------------------------------
	int ntsteps = iters; 
	double dt   = get_timestep(amp, fftw_plan_c2r_pad, fftw_plan_r2c_pad, kx, N, num_osc, k0);

	// If calculating stats or integrating until transient get required iters
	#ifdef __TRANSIENTS
	int trans_iters    = get_transient_iters(amp, fftw_plan_c2r_pad, fftw_plan_r2c_pad, kx, N, num_osc, k0);
	int num_save_steps = ntsteps / SAVE_DATA_STEP; 
	#else 
	int trans_iters    = 0;
	int num_save_steps = ntsteps / SAVE_DATA_STEP + 1; 
	#endif

	// Time variables	
	double t0 = 0.0;
	double T  = t0 + (trans_iters + ntsteps) * dt;


	// ------------------------------
	//  HDF5 File Create
	// ------------------------------
	// Create the HDF5 file handle
	hid_t HDF_Outputfile_handle;


	// create hdf5 handle identifiers for hyperslabing the full evolution data
	hid_t HDF_file_space[4];
	hid_t HDF_data_set[4];
	hid_t HDF_mem_space[4];

	// form the filename of the output file
	#ifdef __STATS
	char output_file_name[128] = "../Data/Output/Stats/Runtime_Data";
	char output_file_data[128];
	sprintf(output_file_data,  "_N[%d]_k0[%d]_ALPHA[%1.3lf]_BETA[%1.3lf]_u0[%s]_ITERS[%d]_TRANS[%d].h5", N, k0, a, b, u0, ntsteps, trans_iters);
	strcat(output_file_name, output_file_data);
	
	// Print file name to screen
	printf("\nOutput File: %s \n\n", output_file_name);
	printf("\n|------------- STATS RUN -------------|\n  Performing transient iterations, no data record...\n\n");
	#else
	char output_file_name[128] = "../Data/Output/Solver/Runtime_Data";
	char output_file_data[128];
	sprintf(output_file_data,  "_N[%d]_k0[%d]_ALPHA[%1.3lf]_BETA[%1.3lf]_u0[%s]_ITERS[%d].h5", N, k0, a, b, u0, ntsteps);
	strcat(output_file_name, output_file_data);
	
	// Print file name to screen
	printf("\nOutput File: %s \n\n", output_file_name);
	#endif	

	// Create complex datatype for hdf5 file if modes are being recorded
	#ifdef __MODES
	hid_t COMPLEX_DATATYPE = create_complex_datatype();
	#else
	hid_t COMPLEX_DATATYPE = -1;
	#endif

	// open output file and create hyperslabbed datasets 
	open_output_create_slabbed_datasets(&HDF_Outputfile_handle, output_file_name, HDF_file_space, HDF_data_set, HDF_mem_space, COMPLEX_DATATYPE, num_save_steps, num_osc, k_range, k1_range);


	

	// ------------------------------
	//  Write Initial Conditions to File
	// ------------------------------
	// Create non chunked data arrays
	double* time_array      = (double* )malloc(sizeof(double) * (num_save_steps));
	#ifdef __TRIADS
	double* phase_order_R   = (double* )malloc(sizeof(double) * (num_save_steps));
	double* phase_order_Phi = (double* )malloc(sizeof(double) * (num_save_steps));	
	#endif

	// Write initial condition if transient iterations are not being performed
	#ifndef __TRANSIENTS
	// Write Initial condition for phases
	write_hyperslab_data(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], H5T_NATIVE_DOUBLE, phi, "phi", num_osc, 0);

	#ifdef __TRIADS
	// compute triads for initial conditions
	triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
	
	// // then write the current modes to this hyperslab
	write_hyperslab_data(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], H5T_NATIVE_DOUBLE, triads, "triads", k_range * k1_range, 0);

	// Write initial order param values
	phase_order_R[0]   = cabs(triad_phase_order);
	phase_order_Phi[0] = carg(triad_phase_order);
	#endif __TRIADS

	#ifdef __MODES
	// Write Initial condition for modes
	write_hyperslab_data(HDF_file_space[2], HDF_data_set[2], HDF_mem_space[2], COMPLEX_DATATYPE, u_z, "u_z", num_osc, 0);
	#endif

	#ifdef __REALSPACE 
	// transform back to Real Space
	fftw_execute_dft_c2r(fftw_plan_c2r, u_z, u);

	// Write Initial condition for real space
	write_hyperslab_data(HDF_file_space[3], HDF_data_set[3], HDF_mem_space[3], H5T_NATIVE_DOUBLE, u, "u", N, 0);
	#endif
	
	// write initial time
	time_array[0] = t0;
	#endif

	
	// ------------------------------
	//  Integration & Looping Params
	// ------------------------------
	int iter = 1;
	double t = 0.0;	
	#ifdef __TRANSIENTS
	int save_data_indx = 0;
	#else
	int save_data_indx = 1;
	#endif

	// ------------------------------
	//  Begin Integration
	// ------------------------------
	while (t < T) {

		// Construct the modes
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * phi[i]);
		}

		// Print Update
		if ((iter > trans_iters) && (iter % print_update == 0)) {
			printf("Iter: %d/%d | t = %4.4lf |\n", iter, ntsteps + trans_iters, t);
		}		

		//////////////
		// STAGES
		//////////////
		/*---------- STAGE 1 ----------*/
		// find RHS first and then update stage
		po_rhs(RK1, u_z_tmp, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, kx, N, num_osc, k0);
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A21 * dt * RK1[i]));
		}

		/*---------- STAGE 2 ----------*/
		// find RHS first and then update stage
		po_rhs(RK2, u_z_tmp, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, kx, N, num_osc, k0);
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A32 * dt * RK2[i]));
		}

		/*---------- STAGE 3 ----------*/
		// find RHS first and then update stage
		po_rhs(RK3, u_z_tmp, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, kx, N, num_osc, k0);
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A43 * dt * RK3[i]));
		}


		/*---------- STAGE 4 ----------*/
		// find RHS first and then update 
		po_rhs(RK4, u_z_tmp, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, kx, N, num_osc, k0);


		//////////////
		// Update
		//////////////
		for (int i = 0; i < num_osc; ++i) {
			phi[i] = phi[i] + (dt * B1) * RK1[i] + (dt * B2) * RK2[i] + (dt * B3) * RK3[i] + (dt * B4) * RK4[i];  
		}

		////////////
		// Print to file
		////////////
		if ((iter > trans_iters) && (iter % save_step == 0)) {
			// Write phases
			write_hyperslab_data(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], H5T_NATIVE_DOUBLE, phi, "phi", num_osc, save_data_indx);

			#ifdef __TRIADS
			// compute triads for initial conditions
			triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
			
			// write triads
			write_hyperslab_data(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], H5T_NATIVE_DOUBLE, triads, "triads", k_range * k1_range, save_data_indx);

			phase_order_R[save_data_indx]   = cabs(triad_phase_order);
			phase_order_Phi[save_data_indx] = carg(triad_phase_order);
			#endif
			#ifdef __MODES || __REALSPACE
			// Construct the modes
			for (int i = 0; i < num_osc; ++i) {
				u_z[i] = amp[i] * cexp(I * phi[i]);
			}
			#endif
			#ifdef __MODES
			// Write Initial condition for modes
			write_hyperslab_data(HDF_file_space[2], HDF_data_set[2], HDF_mem_space[2], COMPLEX_DATATYPE, u_z, "u_z", num_osc, save_data_indx);
			#endif
			#ifdef __REALSPACE 
			// transform back to Real Space
			fftw_execute_dft_c2r(fftw_plan_c2r, u_z, u);

			// Write Initial condition for real space
			write_hyperslab_data(HDF_file_space[3], HDF_data_set[3], HDF_mem_space[3], H5T_NATIVE_DOUBLE, u, "u", N, save_data_indx);
			#endif


			// save time and phase order parameter
			time_array[save_data_indx]      = iter * dt;

			// increment indx for next iteration
			save_data_indx++;
		}
		// Check if transient iterations has been reached
		#ifdef __TRANSIENTS
		if (iter == trans_iters) printf("\n|---- TRANSIENT ITERATIONS COMPLETE ----|\n  Iterations Performed:\t%d\n  Iterations Left:\t%d\n\n", trans_iters, iters);
		#endif

				
		// increment
		t   = iter*dt;
		iter++;
	}
	// ------------------------------
	//  Write 1D Arrays Using HDF5Lite
	// ------------------------------
	hid_t D2 = 2;
	hid_t D2dims[D2];

	// Write amplitudes
	D2dims[0] = 1;
	D2dims[1] = num_osc;
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "Amps", D2, D2dims, H5T_NATIVE_DOUBLE, amp)) < 0){
		printf("\n\n!!Failed to make - Amps - Dataset!!\n\n");
	}
	
	// Wtie time
	D2dims[0] = num_save_steps;
	D2dims[1] = 1;
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "Time", D2, D2dims, H5T_NATIVE_DOUBLE, time_array)) < 0) {
		printf("\n\n!!Failed to make - Time - Dataset!!\n\n");
	}
	
	#ifdef __TRIADS
	// Write Phase Order R
	D2dims[0] = num_save_steps;
	D2dims[1] = 1;
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "PhaseOrderR", D2, D2dims, H5T_NATIVE_DOUBLE, phase_order_R)) < 0) {
		printf("\n\n!!Failed to make - PhaseOrderR - Dataset!!\n\n");
	}
	// Write Phase Order Phi
	D2dims[0] = num_save_steps;
	D2dims[1] = 1;
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "PhaseOrderPhi", D2, D2dims, H5T_NATIVE_DOUBLE, phase_order_Phi)) < 0) {
		printf("\n\n!!Failed to make - PhaseOrderPhi - Dataset!!\n\n");
	}
	#endif


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
	free(time_array);
	fftw_free(u_z);
	fftw_free(RK1);
	fftw_free(RK2);
	fftw_free(RK3);
	fftw_free(RK4);
	fftw_free(u_z_tmp);
	fftw_free(u_z_pad);

	// close HDF5 handles
	H5Sclose( HDF_mem_space[0] );
	H5Dclose( HDF_data_set[0] );
	H5Sclose( HDF_file_space[0] );
	H5Sclose( HDF_mem_space[1] );
	H5Dclose( HDF_data_set[1] );
	H5Sclose( HDF_file_space[1] );
	#ifdef __MODES
	H5Sclose( HDF_mem_space[2] );
	H5Dclose( HDF_data_set[2] );
	H5Sclose( HDF_file_space[2] );
	#endif 
	#ifdef __REALSPACE
	H5Sclose( HDF_mem_space[3] );
	H5Dclose( HDF_data_set[3] );
	H5Sclose( HDF_file_space[3] );
	#endif

	// Close pipeline to output file
	H5Fclose(HDF_Outputfile_handle);
}

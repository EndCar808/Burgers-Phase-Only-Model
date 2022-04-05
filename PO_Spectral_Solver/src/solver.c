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
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_rstat.h>
#include <sys/types.h>
#include <sys/stat.h>


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "utils.h"
#include "stats.h"



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

	// fill amp and phi arrays
	if (strcmp(IC, "ALIGNED") == 0) {
		for (int i = 0; i < num_osc; ++i) {
			// fill the wavenumbers array
			kx[i] = i;

			if(i <= k0) {
				amp[i] = 0.0;
				phi[i] = 0.0;
				u_z[i] = 0.0 + 0.0 * I;
			} else {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
				phi[i] = (M_PI / 2.0) * (1.0 + 1e-10 * pow((double) i, 0.9));		
				u_z[i] = amp[i] * cexp(I * phi[i]);
			}
		}
	} 
	else if (strcmp(IC, "NEW") == 0) {
		for (int i = 0; i < num_osc; ++i) {
			// fill the wavenumbers array
			kx[i] = i;

			if(i <= k0) {
				amp[i] = 0.0;
				phi[i] = 0.0;
				u_z[i] = 0.0 + 0.0 * I;
			} 
			else if (i % 3 == 0){
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
				phi[i] = (M_PI / 2.0) * (1.0 + 1e-10 * pow((double) i, 0.9));	
				u_z[i] = amp[i] * cexp(I * phi[i]);
			} 
			else if (i % 3 == 1) {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
				phi[i] = (M_PI / 6.0) * (1.0 + 1e-10 * pow((double) i, 0.9));	
				u_z[i] = amp[i] * cexp(I * phi[i]);
			}
			else if (i % 3 == 2) {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
				phi[i] = (5.0 * M_PI / 6.0) * (1.0 + 1e-10 * pow((double) i, 0.9));	
				u_z[i] = amp[i] * cexp(I * phi[i]);
			}
		}
	}
	else if (strcmp(IC, "RANDOM") == 0) {
		for (int i = 0; i < num_osc; ++i) {
			// fill the wavenumbers array
			kx[i] = i;

			if(i <= k0) {
				amp[i] = 0.0;
				phi[i] = 0.0;
				u_z[i] = 0.0 + 0.0 * I;
			} else {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );	
				phi[i] = M_PI * ( (double) rand() / (double) RAND_MAX);	
				u_z[i] = amp[i] * cexp(I * phi[i]);
			}
		}
	}
	else if (strcmp(IC, "ZERO") == 0) {
		for (int i = 0; i < num_osc; ++i) {
			// fill the wavenumbers array
			kx[i] = i;
			
			if(i <= k0) {
				amp[i] = 0.0;
				phi[i] = 0.0;
				u_z[i] = 0.0 + 0.0 * I;
			} else {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );	
				phi[i] = 0.0;	
				u_z[i] = amp[i] * cexp(I * phi[i]);
			}
		}
	}
	else if (strcmp(IC, "TEST") == 0) {
		for (int i = 0; i < num_osc; ++i) {
			// fill the wavenumbers array
			kx[i] = i;
			
			if(i <= k0) {
				amp[i] = 0.0;
				phi[i] = 0.0;
				u_z[i] = 0.0 + 0.0 * I;
			} else {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );	
				phi[i] = M_PI / 4.0;	
				u_z[i] = amp[i] * cexp(I * phi[i]);
			}
		}
	}
	else {
		// input file name
		char input_file[256];
		sprintf(input_file, "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Input/Initial_Conditions/FixedPoint_N[%d]_k0[%d]_BETA[%0.3f]_u0[%s].txt", 2*(num_osc - 1),  k0, b, IC); 

		printf("\tInput File: %s\n\n", input_file);
		
		// Open input file
		FILE *in_file  = fopen(input_file, "r"); 
		if (in_file == NULL) {
			fprintf(stderr, "ERROR ("__FILE__":%d) -- Unable to open input file: %s\n", __LINE__, input_file);
			exit(-1);
		}

		for (int i = 0; i < num_osc; ++i) {
			// fill the wavenumbers array
			kx[i] = i;

			// Read in phases
			fscanf(in_file, "%lf", &phi[i]);

			if(i <= k0) {
				amp[i] = 0.0;
				u_z[i] = 0.0 + 0.0 * I;
			} else {
				amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
				u_z[i] = amp[i] * cexp(I * phi[i]);
			}
		}

		fclose(in_file);
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
	char output_dir[512] = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS/";
	char output_dir_tmp[512];
	#ifdef __FXD_PT_SEARCH__
	sprintf(output_dir_tmp,  "FXDPT_SEARCH_k0[%d]_ALPHA[%1.3lf]_BETA[%1.3lf]_u0[%s]", k0, a, b, u0);
	strcat(output_dir, output_dir_tmp);
	#else
	sprintf(output_dir_tmp,  "RESULTS_N[%d]_k0[%d]_ALPHA[%1.3lf]_BETA[%1.3lf]_u0[%s]", N, k0, a, b, u0);
	strcat(output_dir, output_dir_tmp);
	#endif
	
	// Check if output directory exists, if not make directory
	struct stat st = {0};
	if (stat(output_dir, &st) == -1) {
		  mkdir(output_dir, 0700);	  
	}

	// form the filename of the output file	
	char output_file_data[128];
	#ifdef __FXD_PT_SEARCH__
	sprintf(output_file_data,  "/Search_Data_N[%d]_ITERS[%d]_TRANS[%d].h5", N, ntsteps, trans_iters);
	strcpy(output_file_name, output_dir);
	strcat(output_file_name, output_file_data);
	#else
	sprintf(output_file_data,  "/SolverData_ITERS[%d]_TRANS[%d].h5", ntsteps, trans_iters);
	strcpy(output_file_name, output_dir);
	strcat(output_file_name, output_file_data);
	#endif

	#ifdef __STATS	
	// Print file name to screen
	printf("\nOutput File: %s \n\n", output_file_name);
	printf("\n|------------- STATS RUN -------------|\n  Performing transient iterations, no data record...\n\n");
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

void open_output_create_slabbed_datasets(hid_t* file_handle, char* output_file_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, hid_t dtype, int num_t_steps, int num_osc, int k_range, int k1_range) {

	// ------------------------------
	//  Create file
	// ------------------------------	
	// create datafile - H5F_ACC_TRUNC overwrites file if it exists already
	*file_handle = H5Fcreate(output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);


	// ---------------------------------------
	//  Create datasets with hyperslabing
	// ---------------------------------------
	//-----------------------------//
	//---------- PHASES -----------//
	//-----------------------------//
	// create hdf5 dimension arrays for creating the hyperslabs
	const int dimensions = 2;
	hsize_t dims[dimensions];      // array to hold dims of full evolution data
	hsize_t maxdims[dimensions];   // array to hold max dims of full evolution data
	hsize_t chunkdims[dimensions]; // array to hold dims of the hyperslab chunks

	#ifdef __PHASES
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;             // number of timesteps
	dims[1]      = num_osc;                 // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = num_osc;                 // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = num_osc;                 // 1D chunk of size number of modes

	// Create the phases dataset
	create_hdf5_slabbed_dset(file_handle, "Phases", &file_space[0], &data_set[0], &mem_space[0], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	#endif

	
	//-----------------------------//
	//----------- RHS -------------//
	//-----------------------------//
	#ifdef __RHS
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;             // number of timesteps
	dims[1]      = num_osc;                 // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = num_osc;                 // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = num_osc;                 // 1D chunk of size number of modes

	// Create the dataset for the RHS
	create_hdf5_slabbed_dset(file_handle, "RHS", &file_space[4], &data_set[4], &mem_space[4], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
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

    herr_t status = H5Awrite(triads_attr, H5T_NATIVE_INT, triads_dims);

	// close the created attributes obkects
	status = H5Aclose(triads_attr);
    status = H5Sclose(triads_attr_space);	
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

	// Create the dataset for the Modes
	create_hdf5_slabbed_dset(file_handle, "Modes", &file_space[2], &data_set[2], &mem_space[2], dtype, dims, maxdims, chunkdims, dimensions);
	#endif


	//--------------------------------//
	//---------- REALSPACE -----------//
	//--------------------------------//
	#ifdef __REALSPACE
	int NN = 2 * (num_osc - 1);
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;             // number of timesteps
	dims[1]      = NN;                       // number of collocation points
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = NN;                       // number of collocation points
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = NN;                       // number of collocation points

	// Create the dataset for the real space velocity field
	create_hdf5_slabbed_dset(file_handle, "RealSpace", &file_space[3], &data_set[3], &mem_space[3], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	#endif


	//--------------------------------//
	//-------- REALSPACE GRAD --------//
	//--------------------------------//
	#ifdef __GRAD
	int N = 2 * (num_osc - 1);
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;             // number of timesteps
	dims[1]      = N;                       // number of collocation points
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = N;                       // number of collocation points
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = N;                       // number of collocation points


	// Create the dataset for the real space velocity field
	create_hdf5_slabbed_dset(file_handle, "RealSpaceGrad", &file_space[5], &data_set[5], &mem_space[5], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	#endif


	//-------------------------------------------------//
	//---------- SCALE TRIAD ORDER PARMETER -----------//
	//-------------------------------------------------//
	#ifdef __TRIAD_ORDER
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;              // number of timesteps
	dims[1]      = num_osc;                 // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = num_osc;                 // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = num_osc;                 // 1D chunk of size number of modes

	// Create the dataset for the P_k variable - the absolute value of the Adler order parameter
	#ifdef __PHI_K_DOT
	create_hdf5_slabbed_dset(file_handle, "Phi_k_dot", &file_space[6], &data_set[6], &mem_space[6], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	#endif

	// Create the dataset for the Adler scale order parameter  -> P_k(t) e^{i \Phi_k(t)} =  -i (sgnk) e^{i\phi_k}\sum_{k_1} a_{k_1}a_{k-k_1}e^{-i\varphi_{k_1, k - k_1}^{k}}  
	#ifdef __ADLER_PHASE_ORDER
	create_hdf5_slabbed_dset(file_handle, "AdlerScaleOrderParam", &file_space[8], &data_set[8], &mem_space[8], dtype, dims, maxdims, chunkdims, dimensions);
	#endif
	
	// Create the dataset for the phase shift scale dependent scale order parameter -> P_k(t) e^{i \theta_k(t)} =  i (sgnk) \sum_{k_1} a_{k_1}a_{k-k_1}e^{i\varphi_{k_1, k - k_1}^{k}} 
	#ifdef __SCALE_PHASE_ORDER
	create_hdf5_slabbed_dset(file_handle, "PhaseShiftScaleOrderParam", &file_space[9], &data_set[9], &mem_space[9], dtype, dims, maxdims, chunkdims, dimensions);
	#endif

	// Create the dataset for the kuramoto order parameter (in time) for the phase shift theta_k -> T_k(t)e^{i\Theta_k(t)} = 1 / iters \sum_{t}^{iters} e^{i \theta_k(t)}
	#ifdef __THETA_TIME_PHASE_ORDER	
	create_hdf5_slabbed_dset(file_handle, "ThetaTimeScaleOrderParam", &file_space[10], &data_set[10], &mem_space[10], dtype, dims, maxdims, chunkdims, dimensions);
	#endif

	// Create the dataset for the phase locking parameter -> Omega_k = <\dot{Phi}_k> / <F_k>
	#ifdef __OMEGA_K	
	create_hdf5_slabbed_dset(file_handle, "PhiPhaseLocking", &file_space[11], &data_set[11], &mem_space[11], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	#endif

	// Create the dataset for the phase locking parameter -> Omega_k = <\dot{Phi}_k> / <F_k>
	#ifdef __SIN_THETA_K
	create_hdf5_slabbed_dset(file_handle, "SinTheta_k", &file_space[11], &data_set[11], &mem_space[11], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	#endif

	// Create the dataset for the R_k variable - the absolute value of the order parameter in time of the phase-shift theta_k
	// create_hdf5_slabbed_dset(file_handle, "R_k", &file_space[12], &data_set[12], &mem_space[12], H5T_NATIVE_DOUBLE, dims, maxdims, chunkdims, dimensions);
	
	#ifdef __ALT_ORDER_PARAMS
	// Create the dataset for the phase shift scale dependent scale order parameter -> P_k(t) e^{i \theta_k(t)} =  i (sgnk) \sum_{k_1} a_{k_1}a_{k-k_1}e^{i\varphi_{k_1, k - k_1}^{k}} 
	create_hdf5_slabbed_dset(file_handle, "OrderedSyncPhase", &file_space[13], &data_set[13], &mem_space[13], dtype, dims, maxdims, chunkdims, dimensions);

	// Create the dataset for the phase shift scale dependent scale order parameter -> P_k(t) e^{i \theta_k(t)} =  i (sgnk) \sum_{k_1} a_{k_1}a_{k-k_1}e^{i\varphi_{k_1, k - k_1}^{k}} 
	create_hdf5_slabbed_dset(file_handle, "HeavisideSyncPhase", &file_space[14], &data_set[14], &mem_space[14], dtype, dims, maxdims, chunkdims, dimensions);

	// Create the dataset for the phase shift scale dependent scale order parameter -> P_k(t) e^{i \theta_k(t)} =  i (sgnk) \sum_{k_1} a_{k_1}a_{k-k_1}e^{i\varphi_{k_1, k - k_1}^{k}} 
	create_hdf5_slabbed_dset(file_handle, "HeavisideOrderedSyncPhase", &file_space[15], &data_set[15], &mem_space[15], dtype, dims, maxdims, chunkdims, dimensions);
	#endif
	#endif
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
		exit(1);
	}

	// then write the current modes to this hyperslab
	if ((H5Dwrite(data_set, dtype, mem_space, file_space, H5P_DEFAULT, data)) < 0) {
		printf("\n!!Error Writing Slabbed Data!! - For %s at Index: %d \n", data_name, index);
		exit(1);
	}
}

void write_fixed_point(double* phi, double b, char* u0, int N, int num_osc, int k0, int kdim, hid_t* data_set, int save_data_indx) {

	// create fixed point filename
	char output_file[256];
	sprintf(output_file, "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/Input/Initial_Conditions/FixedPoint_N[%d]_k0[%d]_BETA[%0.3f]_u0[%s].txt", 2*(num_osc - 1) + 2,  k0, b, u0); 

	printf("\n\nFixed Point File: %s\n\n", output_file);
	
	// Open output file
	FILE *out_file  = fopen(output_file, "w"); 
	if (out_file == NULL) {
		fprintf(stderr, "ERROR ("__FILE__":%d) -- Unable to open input file: %s\n", __LINE__, output_file);
		exit(-1);
	}

	// write to file including extra random oscillators
	for (int i = 0; i < num_osc + 2; ++i) {
		if (i < num_osc) {
			fprintf(out_file, "%20.16lf\n", phi[i]);
		}
		else {
			fprintf(out_file, "%20.16lf\n", M_PI * ( (double) rand() / (double) RAND_MAX));
		}
	}

	// Close output file
	fclose(out_file);

	// Change size of hdf5 datasets
	hsize_t dims[2];
	dims[0] = save_data_indx;
	dims[1] = num_osc;
	H5Dset_extent(data_set[0], dims);
	#ifdef __MODES
	H5Dset_extent(data_set[2], dims);
	#endif
	#ifdef __RHS
	H5Dset_extent(data_set[4], dims);
	#endif
	#ifdef __REALSPACE
	dims[1] = N;
	H5Dset_extent(data_set[3], dims);
	#endif
	#ifdef __TRIADS
	dims[1] = kdim;
	H5Dset_extent(data_set[1], dims);
	#endif

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
	
	// Initialize variables
	int k1;
	int N = n - 1;
	
	// Compute the convolution for each wavenumber
	for (int kk = 0; kk <= N; ++kk)	{
		if (kk <= 0) {
			convo[kk] = 0.0 + 0.0 * I;
		}
		else {
			for (int k_1 = 0; k_1 <= 2*N; ++k_1)	{
				// Get correct k1 value
				if(k_1 < N) {
					k1 = -N + k_1;
				} else {
					k1 = k_1 - N;
				}

				// Compute the convolution
				if (k1 >= -N + kk) {
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
			// pre_fac = (-I * kx[k]) / (u_z[k]);
			// rhs[k]  = cimag( pre_fac* (u_z_tmp[k] * norm_fac) );
			pre_fac = -kx[k] / cabs(u_z[k]);
			rhs[k]  = pre_fac * creal(cexp(-I * carg(u_z[k])) * (u_z_tmp[k] * norm_fac));
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

void sync_phase(fftw_complex* ordered, fftw_complex* heaviside, fftw_complex* heavi_ordered, double* phi, double* amps, int kmin, int kmax) {

	// Initialize parameters
	int k1;

	for (int k = kmin; k <= kmax; ++k) {
		// Loop over adjusted k1 domain
		for (int kk1 = 0; kk1 <= 2 * kmax - k; ++kk1) {
			// Readjust back to proper k1 
			k1 = kk1 - kmax + k;
			// Consider only valid k1 values
			if(abs(k1) >= kmin && abs(k - k1) >= kmin) {
				// Check if ordered triads was specified or not
				ordered[k]       += amps[abs(k1)] * amps[abs(k - k1)] * cexp(I * (sgn(k - k1) * phi[abs(k1)] + sgn(k1) * phi[abs(k - k1)] - sgn(k1 * (k - k1)) * phi[k]));	
				heaviside[k]     += cexp(I * (sgn(k1) * phi[abs(k1)] + sgn(k - k1) * phi[abs(k - k1)] - phi[k]));				
				heavi_ordered[k] += cexp(I * (sgn(k - k1) * phi[abs(k1)] + sgn(k1) * phi[abs(k - k1)] - sgn(k1 * (k - k1)) * phi[k]));	
			}
		}
	}
}

void amp_normalize(double* norm, double* amp, int num_osc, int k0) {

	// Initialize variables
	int k1;
	int N = num_osc - 1;

	// Compute the sum for each k
	for (int kk = 0; kk <= N; ++kk) {
		if (kk <= k0) {
			norm[kk] = 0.0;
		}
		else {
			for (int k_1 = 0; k_1 <= 2 * N; ++k_1) {
				// Adjust for the correct k1 value
				if (k_1 <= N) {     
					k1 = -N + k_1;
				}
				else  {
					k1 = k_1 - N;
				}
				
				// Compute the convolution
				if (k1 >= - N + kk) {
					norm[kk] +=  amp[abs(k1)] * amp[abs(kk - k1)];
				}
			}
		}
		
	}
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


	// Free memory
	free(tmp_rhs);
	fftw_free(u_z_tmp);

	return trans_iters;
}

double gradient_energy(double* a_k, int* k, int num_osc) {

	// Initialize energy counter
	double energy = 0.0;

	for(int i = 0; i < num_osc; ++i) {
		energy += pow(k[i] * a_k[i], 2);
	}

	return 2.0 * energy / (num_osc - 1);
}

double theoretical_energy(double* a_k, int num_osc) {

	// Initialize energy counter
	double energy = 0.0;

	for(int i = 0; i < num_osc; ++i) {
		energy += pow(a_k[i], 2);
	}

	return 2.0 * energy / (num_osc - 1);
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

void fixed_point_search(int N, int Nmax, int k0, double a, double b, int iters, int save_step, char* u0) {

	// Declare return flag for solver
	int flag;

	// Search for fixed points by looping over N
	for (int i = N; i < Nmax; i+= 2)
	{
		printf("|------------- System Size N = %d -----------|\n", i);
		flag = solver(i, k0, a, b, iters, save_step, u0);
		if (flag == 0) {
			printf("\n\n!!! No fixed point found - Exiting\n");

			// Cleanup and exit
			fftw_cleanup_threads();
			fftw_cleanup();
			exit(-1);
		}
	}
}	




int solver(int N, int k0, double a, double b, int iters, int save_step, int compute_step, char* u0) {

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

	// Triad phases array
	int k_range  = kmax - kmin + 1;
	int k1_range = (int)(k_range/ 2.0);

	// print update every x iterations
	int print_update = (iters >= 10 ) ? (int)((double)iters * 0.1) : 1;

	int indx;
	int tmp;	

	int EXIT_FLAG = 1;
	
	// ------------------------------
	//  Allocate memory
	// ------------------------------
	// wavenumbers
	int* kx = (int* ) malloc(sizeof(int) * num_osc);
	mem_chk(kx, "kx");

	// Oscillator arrays
	double* amp = (double* ) malloc(sizeof(double) * num_osc);
	mem_chk(amp, "amp");
	double* phi = (double* ) malloc(sizeof(double) * num_osc);
	mem_chk(phi, "phi");
	#ifdef __FXD_PT_SEARCH__
	double* rhs_prev = (double* ) malloc(sizeof(double) * num_osc);
	mem_chk(rhs_prev, "rhs_prev");
	for (int i = 0; i < num_osc; ++i) rhs_prev[i] = 0.0;
	double sum;
	int iters_at_fxd_pt = 0;
	#endif

	// modes array
	fftw_complex* u_z = (fftw_complex* ) fftw_malloc(num_osc * sizeof(fftw_complex));
	mem_chk(u_z, "u_z"); 

	// padded solution arrays
	double* u_pad = (double* ) malloc(M * sizeof(double));
	mem_chk(u_pad, "u_pad");
	fftw_complex* u_z_pad = (fftw_complex* ) fftw_malloc((2 * num_osc - 1) * sizeof(fftw_complex));
	mem_chk(u_z_pad, "u_z_pad");

	#if defined(__REALSPACE) || defined(__REALSPACE_STATS)
	double* u = (double* ) malloc(N * sizeof(double));
	mem_chk(u, "u");
	for (int i = 0; i < N; ++i) u[i] = 0.0;
	#endif
	#if defined(__GRAD) || defined(__REALSPACE_STATS)
	fftw_complex* u_z_grad = (fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(u_z_grad, "u_z_grad");
	double* u_grad = (double* ) malloc(sizeof(double) * N);
	mem_chk(u_grad, "u_grad");

	for (int i = 0; i < N; ++i) {
		u_grad[i] = 0.0;
		if (i < num_osc){
			u_z_grad[i] = 0.0 + 0.0 * I;
		}
	}
	#endif
	#ifdef __REALSPACE_STATS
	// Initialize vel inc histogram & run stats
	int num_r_inc = 2;
	gsl_histogram* vel_inc_hist[num_r_inc + 1];
	gsl_rstat_workspace* vel_inc_stats[num_r_inc + 1];

	for (int i = 0; i < num_r_inc + 1; ++i) {
	 	vel_inc_hist[i]  = gsl_histogram_alloc(NBIN_VELINC);
	 	vel_inc_stats[i] = gsl_rstat_alloc();
	} 
	
	// Second moment variables
	double vel_sec_mnt;
	double grad_sec_mnt;	

	// Stats counter
	int num_stats = 0;
	
	// stucture function array
	int max_p     = 6;
	double *str_func = NULL;

	#ifdef __STR_FUNCS	
	str_func = (double *)malloc(sizeof(double) * (max_p - 2 + 1) * (num_osc - 1));
	mem_chk(str_func, "str_func");
	for (int i = 0; i < (max_p - 2 + 1); ++i) {
		for (int j = 0; j < (num_osc - 1); ++j) {
			str_func[i * (num_osc - 1) + j] = 0.0;
		}
	}
	#endif
	#endif
	#if  defined(__TRIADS) || defined(__TRIAD_STATS)
	// Initialize tirad array
	double* triads = (double* )malloc(k_range * k1_range * sizeof(double));
	mem_chk(triads, "triads");
	// initialize triad array to handle empty elements
	for (int i = 0; i < k_range; ++i) {
		for (int j = 0; j < k1_range; ++j) {
			triads[i * k1_range + j] = -10.0;
		}
	}

	// Initialize phase order variable
	fftw_complex triad_phase_order = 0.0 + I * 0.0;

	#ifdef __TRIAD_STATS
	int num_triad_stats = 0;
	// Initialize triad centroid arrays
	fftw_complex* triad_centroid = (fftw_complex* )fftw_malloc(k_range * k1_range * sizeof(fftw_complex));
	mem_chk(triad_centroid, "triad_centroid");
	double* triad_cent_R         = (double* )malloc(k_range * k1_range * sizeof(double));
	mem_chk(triad_cent_R, "triad_cent_R");
	double* triad_cent_Phi       = (double* )malloc(k_range * k1_range * sizeof(double));
	mem_chk(triad_cent_Phi, "triad_cent_Phi");

	// Set to -10 to make handling empty elements easier later
	for (int i = 0; i < k_range; ++i) {
		for (int j = 0; j < k1_range; ++j) {
			triad_centroid[i * k1_range + j] = -10.0 -10.0 * I;
			triad_cent_R[i * k1_range + j]   = -10.0;
			triad_cent_Phi[i * k1_range + j] = -10.0;
		}
	}
	#endif
	#endif
	#ifdef __TRIAD_ORDER
	// Allocate scale dependent order paramter arrays
	fftw_complex* adler_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(adler_order_k, "adler_order_k");
	fftw_complex* phase_shift_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(phase_shift_order_k, "phase_shift_order_k");
	fftw_complex* theta_time_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(theta_time_order_k, "theta_time_order_k");
	fftw_complex* tmp_time_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(tmp_time_order_k, "tmp_time_order_k");

	#ifdef __ALT_ORDER_PARAMS
	fftw_complex* ordered_sync_phase = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(ordered_sync_phase, "ordered_sync_phase");
	fftw_complex* heaviside_sync_phase = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(heaviside_sync_phase, "heaviside_sync_phase");
	fftw_complex* heaviside_ordered_sync_phase = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(heaviside_ordered_sync_phase, "heaviside_ordered_sync_phase");
	fftw_complex* tmp_ordered_time_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(tmp_ordered_time_order_k, "tmp_ordered_time_order_k");
	fftw_complex* tmp_heavi_time_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(tmp_heavi_time_order_k, "tmp_heavi_time_order_k");
	fftw_complex* tmp_heavi_order_time_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(tmp_heavi_order_time_order_k, "tmp_heavi_order_time_order_k");
	fftw_complex* ordered_time_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(ordered_time_order_k, "ordered_time_order_k");
	fftw_complex* heaviside_time_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(heaviside_time_order_k, "heaviside_time_order_k");
	fftw_complex* heavi_order_time_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(heavi_order_time_order_k, "heavi_order_time_order_k");
	#endif

	// Allocate arrays needed to compute the phase order parameters
	fftw_complex* conv = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(conv, "conv");
	double* amp_norm = (double* )malloc(sizeof(double) * num_osc);
	mem_chk(amp_norm, "amp_norm");

	// Allocate memory for the phase locking quantities
	double* F_k_avg   = (double* )malloc(sizeof(double) * num_osc);
	mem_chk(F_k_avg, "F_k_avg");
	double* Phi_k_dot_avg = (double* )malloc(sizeof(double) * num_osc); 
	mem_chk(Phi_k_dot_avg, "Phi_k_dot_avg");
	double* Phi_k_dot = (double* )malloc(sizeof(double) * num_osc); 
	mem_chk(Phi_k_dot, "Phi_k_dot");
	double* Phi_k_last = (double* )malloc(sizeof(double) * num_osc); 
	mem_chk(Phi_k_last, "Phi_k_last");
	double* Omega_k = (double* )malloc(sizeof(double) * num_osc); 
	mem_chk(Omega_k, "Omega_k");

	// Allocate memory for the real order quantities
	double* sin_theta_k_avg = (double* )malloc(sizeof(double) * num_osc); 
	mem_chk(sin_theta_k_avg, "sin_theta_k_avg");
	double* sin_theta_k = (double* )malloc(sizeof(double) * num_osc); 
	mem_chk(sin_theta_k, "sin_theta_k");
	double* P_k = (double* )malloc(sizeof(double) * num_osc); 
	mem_chk(P_k, "P_k");
	double* R_k = (double* )malloc(sizeof(double) * num_osc); 
	mem_chk(R_k, "R_k");

	// Initialize counter for taking running averages
	int t_count = 1;

	// Initialize the arrays
	for (int i = 0; i < num_osc; ++i) {
		// Order params
		adler_order_k[i]       = 0.0 + 0.0 * I;
		phase_shift_order_k[i] = 0.0 + 0.0 * I;
		theta_time_order_k[i]  = 0.0 + 0.0 * I;

		// Auxillary arrays
		tmp_time_order_k[i]    = 0.0 + 0.0 * I;
		conv[i]                = 0.0 + 0.0 * I;
		amp_norm[i]            = 0.0;

		// Phase-locking arrays
		F_k_avg[i]       = 0.0;
		Omega_k[i]       = 0.0;
		Phi_k_last[i]	 = 0.0;
		Phi_k_dot_avg[i] = 0.0;

		P_k[i]             = 0.0;
		R_k[i]             = 0.0;
		sin_theta_k[i]     = 0.0;
		sin_theta_k_avg[i] = 0.0;
	}
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
	mem_chk(RK1, "RK1");
	mem_chk(RK2, "RK2");
	mem_chk(RK3, "RK3");
	mem_chk(RK4, "RK4");

	// temporary memory to store stages
	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");
	

	// ------------------------------
	//  Create FFTW plans
	// ------------------------------
	// create fftw3 plans objects - ensure no overwriting - fill arrays after
	fftw_plan fftw_plan_r2c_pad, fftw_plan_c2r_pad;
	fftw_plan_r2c_pad = fftw_plan_dft_r2c_1d(M, u_pad, u_z_pad, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r_pad = fftw_plan_dft_c2r_1d(M, u_z_pad, u_pad, FFTW_PRESERVE_INPUT);
	#if defined(__REALSPACE) || defined(__REALSPACE_STATS) || defined(__GRAD)
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
	int num_save_steps = ntsteps / save_step; 
	#else 
	int trans_iters    = 0;
	int num_save_steps = ntsteps / save_step + 1; 
	#endif

	// Time variables	
	double t0 = 0.0;
	double T  = t0 + (trans_iters + ntsteps) * dt;

	
	// ------------------------------
	//  HDF5 File Create
	// ------------------------------
	// Create the HDF5 file handle
	hid_t HDF_Outputfile_handle;

	// HDF5 error handling variable
	herr_t status;


	// create hdf5 handle identifiers for hyperslabing the full evolution data
	hid_t HDF_file_space[13];
	hid_t HDF_data_set[13];
	hid_t HDF_mem_space[13];

	// get output file name
	char output_file_name[512];
	get_output_file_name(output_file_name, N, k0, a, b, u0, ntsteps, trans_iters);

	// Create complex datatype for hdf5 file if modes are being recorded
	#if defined(__MODES) || defined(__TRIAD_STATS) || defined(__TRIAD_ORDER)
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
	#ifdef __TIME
	double* time_array      = (double* )malloc(sizeof(double) * (num_save_steps));
	mem_chk(time_array, "time_array");
	#endif
	#ifdef __TRIADS
	double* phase_order_R   = (double* )malloc(sizeof(double) * (num_save_steps));
	double* phase_order_Phi = (double* )malloc(sizeof(double) * (num_save_steps));
	mem_chk(phase_order_R, "phase_order_R");
	mem_chk(phase_order_Phi, "phase_order_Phi");	
	#endif

	// Write initial condition if transient iterations are not being performed
	#ifndef __TRANSIENTS
	#ifdef __PHASES
	// Write Initial condition for phases
	write_hyperslab_data(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], H5T_NATIVE_DOUBLE, phi, "phi", num_osc, 0);
	#endif

	#ifdef __TRIADS
	// compute triads for initial conditions
	triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
	
	// then write the current modes to this hyperslab
	write_hyperslab_data(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], H5T_NATIVE_DOUBLE, triads, "triads", k_range * k1_range, 0);

	// Write initial order param values
	phase_order_R[0]   = cabs(triad_phase_order);
	phase_order_Phi[0] = carg(triad_phase_order);
	#endif

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
	
	#ifdef __TIME
	// write initial time
	time_array[0] = t0;
	#endif
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

	clock_t begin;

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


		/////////////////////////
		// Compute Runtime Data
		/////////////////////////
		if ((iter > trans_iters) && (iter % compute_step == 0)) {		
						 
			#ifdef __TRIAD_ORDER
			// Get the normalization constant
			if (save_data_indx == 0) {
				amp_normalize(amp_norm, amp, num_osc, k0);
			}

			// Get the convolution
			conv_2N_pad(conv, u_z, &fftw_plan_r2c_pad, &fftw_plan_c2r_pad, N, num_osc, k0);

			#ifdef __ALT_ORDER_PARAMS
			// Calculate alternative sync order params
			sync_phase(ordered_sync_phase, heaviside_sync_phase, heaviside_ordered_sync_phase, phi, amp, kmin, kmax);
			#endif

			// compute scale dependent phase order parameter
			for (int i = kmin; i < num_osc; ++i) {
				// Proposed scale dependent Kuramoto order parameters
				adler_order_k[i] = -I * (cexp(I * 2.0 *  phi[i]) * conj(conv[i])); // / amp_norm[i]);

				// P_k
				P_k[i] += cabs(adler_order_k[i]);

				// Scale dependent Phase shift order parameter
				phase_shift_order_k[i] = I * (conv[i] * cexp(-I * phi[i])); // / amp_norm[i]);
				
				// The phaseshift parameters - theta_k
				sin_theta_k[i] = sin(carg(phase_shift_order_k[i]));
				sin_theta_k_avg[i] += sin_theta_k[i]; 

				// Update the temporary time order parameters
				tmp_time_order_k[i]             += cexp(I * carg(phase_shift_order_k[i]));
				#ifdef __ALT_ORDER_PARAMS
				tmp_ordered_time_order_k[i]     += cexp(I * carg(ordered_sync_phase[i]));
				tmp_heavi_time_order_k[i]       += cexp(I * carg(heaviside_sync_phase[i]));
				tmp_heavi_order_time_order_k[i] += cexp(I * carg(heaviside_ordered_sync_phase[i]));
				#endif
				// Calculate the Kuramoto order parameter in time
				theta_time_order_k[i]       = tmp_time_order_k[i] / t_count;
				#ifdef __ALT_ORDER_PARAMS
				ordered_time_order_k[i]     = tmp_ordered_time_order_k[i] / t_count;
				heaviside_time_order_k[i]   = tmp_heavi_time_order_k[i] / t_count;
				heavi_order_time_order_k[i] = tmp_heavi_order_time_order_k[i] / t_count;
				#endif

				// The synchronization parameter of the order param in time - R_k
				R_k[i] += cabs(theta_time_order_k[i]);

				// Adler parameters
				F_k_avg[i] += ((double)i / (2.0 * amp[i])) * cabs(adler_order_k[i]);
				if (save_data_indx > 0) {
					// Finite difference for \dot{\Phi}_k = arg(exp^{i \Phi_k(t2)}exp^{-i \Phi_k(t1)}) -
					Phi_k_dot[i] = carg(adler_order_k[i] * cexp(-I * Phi_k_last[i])) / (compute_step * dt);
					Phi_k_dot_avg[i] += Phi_k_dot[i];

					// Locking -> omega_k = <\dot{\Phi}_k> / <F_k>
					Omega_k[i] = (Phi_k_dot_avg[i] / t_count) / (F_k_avg[i] / t_count);
				}
			}

			// Update Phi_k_last for next iteration
			for (int i = 0; i < num_osc; ++i) {
				Phi_k_last[i] = carg(adler_order_k[i]);
			}

			// increment the counter			
			t_count++;
			#endif

			#ifdef __REALSPACE_STATS			
			// If first non-transient iteration - set bin edges
			if ((trans_iters != 0) && (save_data_indx == 0)) {
				// Compute the second moments of the velocity and gradient fields
				vel_sec_mnt  = sqrt(theoretical_energy(amp, num_osc));
				grad_sec_mnt = sqrt(gradient_energy(amp, kx, num_osc));				
				
				// Initialize the histogram bins for the PDFs
				gsl_set_vel_inc_hist_bin_ranges(vel_inc_hist, u, u_grad, vel_sec_mnt, grad_sec_mnt, num_osc);
			}
			
			// Compute velocity increment PDFs, stats, structure funcs etc.
			gsl_compute_real_space_stats(vel_inc_hist, vel_inc_stats, str_func, u, u_grad, vel_sec_mnt, grad_sec_mnt, num_osc, max_p);

			// Increment the stats counter
			num_stats++;
			#endif
		}


		////////////////////////
		// Print to file
		////////////////////////
		if ((iter > trans_iters) && (iter % save_step == 0)) {		
						
			#ifdef __PHASES
			// Write phases
			write_hyperslab_data(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], H5T_NATIVE_DOUBLE, phi, "phi", num_osc, save_data_indx);
			#endif

			#if defined(__TRIADS) || defined(__TRIAD_STATS)
			// compute triads for initial conditions
			triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
	
			#ifdef __TRIAD_STATS
			for (int k = kmin; k <= kmax; ++k) {
				tmp = (k - kmin) * (int) ((kmax - kmin + 1) / 2.0);
				for (int k1 = kmin; k1 <= (int) (k / 2.0); ++k1) {
					indx = tmp + (k1 - kmin);
					// update triad centroid
					triad_centroid[indx] += cexp(I * triads[indx]);
				}
			}
			// Increment triad stats counter
			num_triad_stats++;
			#endif
			#endif
			
			#ifdef __TRIADS
			// write triads
			write_hyperslab_data(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], H5T_NATIVE_DOUBLE, triads, "triads", k_range * k1_range, save_data_indx);

			phase_order_R[save_data_indx]   = cabs(triad_phase_order);
			phase_order_Phi[save_data_indx] = carg(triad_phase_order);
			#endif

			#if defined(__MODES) || defined(__GRAD) || defined(__REALSPACE) || defined(__REALSPACE_STATS) || defined(__TRIAD_ORDER)
			// Construct the modes
			for (int i = 0; i < num_osc; ++i) {
				u_z[i] = amp[i] * cexp(I * phi[i]);

				#if defined(__GRAD) || defined(__REALSPACE_STATS)
				u_z_grad[i] = I * kx[i] * u_z[i];
				#endif
			}
			#endif

			#ifdef __MODES
			// Write current modes
			write_hyperslab_data(HDF_file_space[2], HDF_data_set[2], HDF_mem_space[2], COMPLEX_DATATYPE, u_z, "u_z", num_osc, save_data_indx);
			#endif

			#ifdef __RHS
			// Write RHS
			write_hyperslab_data(HDF_file_space[4], HDF_data_set[4], HDF_mem_space[4], H5T_NATIVE_DOUBLE, RK1, "RHS", num_osc, save_data_indx);
			#endif

			#if defined(__REALSPACE) || defined(__REALSPACE_STATS) 
			// transform back to Real Space
			fftw_execute_dft_c2r(fftw_plan_c2r, u_z, u);			
			for (int i = 0; i < N; ++i)	{
				u[i] /= sqrt((double) N);  // Normalize inverse transfom
			}
			#endif

			#if defined(__GRAD) || defined(__REALSPACE_STATS) 
			// transform back to Real Space
			fftw_execute_dft_c2r(fftw_plan_c2r, u_z_grad, u_grad);
			for (int i = 0; i < N; ++i)	{
				u_grad[i] /= sqrt((double) N);  // Normalize inverse transfom
			}
			#endif

			#ifdef __TRIAD_ORDER
			// Write the scale order parameters to file
			#ifdef __ADLER_PHASE_ORDER
			write_hyperslab_data(HDF_file_space[8], HDF_data_set[8], HDF_mem_space[8], COMPLEX_DATATYPE, adler_order_k, "AdlerScaleOrderParam", num_osc, save_data_indx);
			#endif
			#ifdef __SCALE_PHASE_ORDER
			write_hyperslab_data(HDF_file_space[9], HDF_data_set[9], HDF_mem_space[9], COMPLEX_DATATYPE, phase_shift_order_k, "PhaseShiftScaleOrderParam", num_osc, save_data_indx);
			#endif
			#ifdef __THETA_TIME_PHASE_ORDER
			write_hyperslab_data(HDF_file_space[10], HDF_data_set[10], HDF_mem_space[10], COMPLEX_DATATYPE, theta_time_order_k, "ThetaTimeScaleOrderParam", num_osc, save_data_indx);
			#endif
			#ifdef __OMEGA_K
			write_hyperslab_data(HDF_file_space[11], HDF_data_set[11], HDF_mem_space[11], H5T_NATIVE_DOUBLE, Omega_k, "PhiPhaseLocking", num_osc, save_data_indx);
			#endif
			#ifdef __SIN_THETA_K
			write_hyperslab_data(HDF_file_space[11], HDF_data_set[11], HDF_mem_space[11], H5T_NATIVE_DOUBLE, sin_theta_k, "SinTheta_k", num_osc, save_data_indx);
			#endif
			#ifdef __PHI_K_DOT
			write_hyperslab_data(HDF_file_space[6], HDF_data_set[6], HDF_mem_space[6], H5T_NATIVE_DOUBLE, Phi_k_dot, "Phi_k_dot", num_osc, save_data_indx);
			#endif
			
			// write_hyperslab_data(HDF_file_space[12], HDF_data_set[12], HDF_mem_space[12], H5T_NATIVE_DOUBLE, R_k, "R_k", num_osc, save_data_indx);
			#endif

			#ifdef __GRAD
			// Write real space
			write_hyperslab_data(HDF_file_space[5], HDF_data_set[5], HDF_mem_space[5], H5T_NATIVE_DOUBLE, u_grad, "u_grad", N, save_data_indx);
			#endif
			#ifdef __REALSPACE
			// Write real space
			write_hyperslab_data(HDF_file_space[3], HDF_data_set[3], HDF_mem_space[3], H5T_NATIVE_DOUBLE, u, "u", N, save_data_indx);
			#endif

			#ifdef __TIME
			// save time and phase order parameter
			time_array[save_data_indx]  = iter * dt;
			#endif

			#ifdef __FXD_PT_SEARCH__
			// check if system has fallen into a fixed point
			if (iters_at_fxd_pt > 5) {
				printf("\n\nTrajectory fallen into fixed point at iteration: %d\n", iter);

				// Write fixed point phases to file & change size of hdf5 datasets
				write_fixed_point(phi, b, u0, N, num_osc, k0, k_range * k1_range, HDF_data_set, save_data_indx);

				// change number of save steps and break loop
				num_save_steps = save_data_indx;
				break;
			}
			#endif 

			// increment indx for next iteration
			save_data_indx++;
		}
		// Check if transient iterations has been reached
		#ifdef __TRANSIENTS
		if (iter == trans_iters) printf("\n|---- TRANSIENT ITERATIONS COMPLETE ----|\n  Iterations Performed:\t%d\n  Iterations Left:\t%d\n\n", trans_iters, iters);
		#endif


		#ifdef __FXD_PT_SEARCH__
		// check if system has fallen into a fixed point
		if (iter > 1) {
			sum = 0.0;
			for (int i = 0; i < num_osc; ++i) {
				sum += fabs(RK1[i] - rhs_prev[i]);
			}

			if (sum <= 0.0) {
				iters_at_fxd_pt++;
			}
		}
		// update for next iteration
		memcpy(rhs_prev, RK1, num_osc * sizeof(double));
		#endif

				
		// increment
		t   = iter*dt;
		iter++;
	}
	// ------------------------------
	//  End Integration
	// ------------------------------


	// ------------------------------
	//  Write 1D Arrays Using HDF5Lite
	// ------------------------------
	const hsize_t D1 = 1;
	hsize_t D1dims[D1];

	#ifdef __AMPS
	// Write amplitudes
	D1dims[0] = num_osc;
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "Amps", D1, D1dims, H5T_NATIVE_DOUBLE, amp)) < 0) {
		printf("\n\n!!Failed to make - Amps - Dataset!!\n\n");
	}
	#endif
	
	#ifdef __TIME
	// Wtie time
	D1dims[0] = num_save_steps;
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "Time", D1, D1dims, H5T_NATIVE_DOUBLE, time_array)) < 0) {
		printf("\n\n!!Failed to make - Time - Dataset!!\n\n");
	}
	#endif
	
	#ifdef __TRIADS
	// Write Phase Order R
	D1dims[0] = num_save_steps;
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "PhaseOrderR", D1, D1dims, H5T_NATIVE_DOUBLE, phase_order_R)) < 0) {
		printf("\n\n!!Failed to make - PhaseOrderR - Dataset!!\n\n");
	}
	// Write Phase Order Phi
	D1dims[0] = num_save_steps;
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "PhaseOrderPhi", D1, D1dims, H5T_NATIVE_DOUBLE, phase_order_Phi)) < 0) {
		printf("\n\n!!Failed to make - PhaseOrderPhi - Dataset!!\n\n");
	}
	#endif

	#ifdef __TRIAD_ORDER
	// Dimension of these dsets
	D1dims[0] = num_osc;
	
	// // Write phase locking
		// if ( (H5LTmake_dataset(HDF_Outputfile_handle, "PhiPhaseLocking", D1, D1dims, H5T_NATIVE_DOUBLE, Omega_k)) < 0) {
	// 	printf("\n\n!!Failed to make - PhiPhaseLocking - Dataset!!\n\n");
	// }
	// Normalize P_k and R_k
	for (int i = 0; i < num_osc; ++i) {
		P_k[i] /= (t_count - 1);
		R_k[i] /= (t_count - 1);
		sin_theta_k_avg[i] /= (t_count -1);
	}
	
	// Write P_k averaged over time
	#ifdef __P_K_AVG
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "P_k_avg", D1, D1dims, H5T_NATIVE_DOUBLE, P_k)) < 0) {
		printf("\n\n!!Failed to make - P_k_avg - Dataset!!\n\n");
	}
	#endif
	
	// Write sin(Theta_k) averaged over time
	#ifdef __SIN_THETA_K_AVG
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "SinTheta_k_avg", D1, D1dims, H5T_NATIVE_DOUBLE, sin_theta_k_avg)) < 0) {
		printf("\n\n!!Failed to make - SinTheta_k_avg - Dataset!!\n\n");
	}
	#endif
	
	// Write R_k averaged over time
	#ifdef __R_K_AVG
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "R_k_avg", D1, D1dims, H5T_NATIVE_DOUBLE, R_k)) < 0) {
		printf("\n\n!!Failed to make - R_k_avg - Dataset!!\n\n");
	}
	#endif

	// Write the alternative order paramters to file
	#ifdef __ALT_ORDER_PARAMS
	double* tmp_array = (double* )malloc(sizeof(double) * num_osc);
	for (int i = 0; i < num_osc; ++i) {
		tmp_array[i] = cabs(ordered_time_order_k[i]);
	}
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "OrderedSyncPhase", D1, D1dims, H5T_NATIVE_DOUBLE, tmp_array)) < 0) {
		printf("\n\n!!Failed to make - OrderedSyncPhase - Dataset!!\n\n");
	}
	for (int i = 0; i < num_osc; ++i) {
		tmp_array[i] = cabs(heaviside_time_order_k[i]);
	}
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "HeavisideSyncPhase", D1, D1dims, H5T_NATIVE_DOUBLE, tmp_array)) < 0) {
		printf("\n\n!!Failed to make - HeavisideSyncPhase - Dataset!!\n\n");
	}
	for (int i = 0; i < num_osc; ++i) {
		tmp_array[i] = cabs(heavi_order_time_order_k[i]);
	}
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "HeavisideOrderedSyncPhase", D1, D1dims, H5T_NATIVE_DOUBLE, tmp_array)) < 0) {
		printf("\n\n!!Failed to make - HeavisideOrderedSyncPhase - Dataset!!\n\n");
	}
	free(tmp_array);
	#endif
	#endif
	
	#ifdef __REALSPACE_STATS
	char dset_name_binedges[256];
	char dset_name_bincounts[256];
	double vel_inc_stats_data[num_r_inc + 1][4];
	for (int i = 0; i < num_r_inc + 1; ++i) {
		// Set dataset names
		if (i < num_r_inc) {
			sprintf(dset_name_binedges, "VelInc[%d]_BinEdges", i);
			sprintf(dset_name_bincounts, "VelInc[%d]_BinCounts", i);
		} 
		else {
			strcpy(dset_name_binedges, "VelGrad_BinEdges");
			strcpy(dset_name_bincounts, "VelGrad_BinCounts");
		}
		
		// Write the bin edges
		D1dims[0] = NBIN_VELINC + 1;
		if ( (H5LTmake_dataset(HDF_Outputfile_handle, dset_name_binedges, D1, D1dims, H5T_NATIVE_DOUBLE, vel_inc_hist[i]->range)) < 0) {
			printf("\n\n!!Failed to make - %s - Dataset!!\n\n", dset_name_binedges);
		}
		// Write the bin counts
		D1dims[0] = NBIN_VELINC;
		if ( (H5LTmake_dataset(HDF_Outputfile_handle, dset_name_bincounts, D1, D1dims, H5T_NATIVE_DOUBLE, vel_inc_hist[i]->bin)) < 0) {
			printf("\n\n!!Failed to make - %s - Dataset!!\n\n", dset_name_bincounts);
		}		
		
		// Collect Stats
		vel_inc_stats_data[i][0] = gsl_rstat_mean(vel_inc_stats[i]); 
		vel_inc_stats_data[i][1] = gsl_rstat_variance(vel_inc_stats[i]); 
		vel_inc_stats_data[i][2] = gsl_rstat_skew(vel_inc_stats[i]); 
		vel_inc_stats_data[i][3] = gsl_rstat_kurtosis(vel_inc_stats[i]); 				
	}	

	// Write stats
	const hsize_t D2 = 2;
	hsize_t D2dims[D2];
	D2dims[0] = num_r_inc + 1;
	D2dims[1] = 4;
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "VelIncStats", D2, D2dims, H5T_NATIVE_DOUBLE, vel_inc_stats_data)) < 0) {
		printf("\n\n!!Failed to make - %s - Dataset!!\n\n", "VelIncStats");
	}	

	#ifdef __STR_FUNCS
	// Normalize and write structure functions
	double str_funcs[max_p - 2 + 1][num_osc - 1];
	for (int p = 2; p <= max_p; ++p) {
		for (int r = 0; r < num_osc - 1; ++r) {
			str_funcs[p - 2][r]     = str_func[(p - 2) * (num_osc - 1) + r] / num_stats;
		}
	}
	D2dims[0] = (max_p - 2 + 1);
	D2dims[1] = (num_osc - 1);
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "StructureFuncs", D2, D2dims, H5T_NATIVE_DOUBLE, str_funcs)) < 0) {
		printf("\n\n!!Failed to make - %s - Dataset!!\n\n", "StructureFuncs");
	}	
	#endif 
	#endif
	#ifdef __TRIAD_STATS
	D1dims[0] = k_range * k1_range;

	for (int k = kmin; k <= kmax; ++k) {
		tmp = (k - kmin) * (int) ((kmax - kmin + 1) / 2.0);
		for (int k1 = kmin; k1 <= (int) (k / 2.0); ++k1) {
			indx = tmp + (k1 - kmin);

			triad_centroid[indx] /= num_triad_stats;
			triad_cent_R[indx]   = cabs(triad_centroid[indx]);
			triad_cent_Phi[indx] = carg(triad_centroid[indx]);
		}
	}
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "TriadCentroid_R", D1, D1dims, H5T_NATIVE_DOUBLE, triad_cent_R)) < 0) {
		printf("\n\n!!Failed to make - TriadCentroid_R - Dataset!!\n\n");
	}
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "TriadCentroid_Phi", D1, D1dims, H5T_NATIVE_DOUBLE, triad_cent_Phi)) < 0) {
		printf("\n\n!!Failed to make - TriadCentroid_Phi - Dataset!!\n\n");
	}

	// Create dataspace
    hid_t triad_cent_dspace = H5Screate_simple(D1, D1dims, NULL);

    // Create dataset
	hid_t triad_cent_dset = H5Dcreate2(HDF_Outputfile_handle, "TriadCentroid", COMPLEX_DATATYPE, triad_cent_dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// Write dataset
	status H5Dwrite(triad_cent_dset, COMPLEX_DATATYPE, H5S_ALL, H5S_ALL, H5P_DEFAULT, triad_centroid);


	// Create attribute data for the triad_centroid dimensions
	hid_t triad_cent, triad_cent_space;

	hsize_t t_adims[1];
	t_adims[0] = 2;
	t_adims[1] = 2;

	triad_cent_space = H5Screate_simple(2, t_adims, NULL);

	triad_cent = H5Acreate(triad_cent_dset, "Triad_Dims", H5T_NATIVE_INT, triad_cent_space, H5P_DEFAULT, H5P_DEFAULT);

	int triad_cent_dims[2];
	triad_cent_dims[0] = k_range;
	triad_cent_dims[1] = k1_range;

    status = H5Awrite(triad_cent, H5T_NATIVE_INT, triad_cent_dims);

	// close the created property list
	status = H5Aclose(triad_cent);
    status = H5Sclose(triad_cent_space);
    // Close datasets and spaces
	status = H5Dclose(triad_cent_dset);
	status = H5Sclose(triad_cent_dspace);
	#endif


	
	// ------------------------------
	//  Clean Up
	// ------------------------------
	// destroy fftw plans
	#if defined(__REALSPACE) || defined(__REALSPACE_STATS) || defined(__GRAD)
	fftw_destroy_plan(fftw_plan_r2c);
	fftw_destroy_plan(fftw_plan_c2r);
	#endif
	fftw_destroy_plan(fftw_plan_r2c_pad);
	fftw_destroy_plan(fftw_plan_c2r_pad);


	#ifdef __REALSPACE_STATS
	// Free gsl histogram and running stat structs
	for (int i = 0; i < num_r_inc + 1; ++i){
		gsl_histogram_free(vel_inc_hist[i]);
		gsl_rstat_free(vel_inc_stats[i]);
	}
	#endif


	// free memory
	free(kx);
	free(amp);
	free(phi);
	free(u_pad);
	#ifdef __FXD_PT_SEARCH__
	free(rhs_prev);
	#endif
	#if defined(__REALSPACE) || defined(__REALSPACE_STATS)
	free(u);
	#endif
	#if defined(__GRAD) || defined(__REALSPACE_STATS)
	fftw_free(u_z_grad);
	free(u_grad);
	#endif
	#if defined(__TRIADS) || defined(__TRIAD_STATS)
	free(triads);	
	#endif
	#ifdef __TRIADS
	free(phase_order_Phi);
	free(phase_order_R);
	#endif	
	#ifdef __TRIAD_STATS
	free(triad_centroid);	
	free(triad_cent_R);
	free(triad_cent_Phi);
	#endif
	#ifdef __TRIAD_ORDER
	fftw_free(adler_order_k);
	fftw_free(phase_shift_order_k);	
	fftw_free(tmp_time_order_k);
	fftw_free(theta_time_order_k);	
	fftw_free(conv);

	#ifdef __ALT_ORDER_PARAMS
	fftw_free(ordered_time_order_k);
	fftw_free(heaviside_time_order_k);
	fftw_free(heavi_order_time_order_k);
	fftw_free(ordered_sync_phase);
	fftw_free(heaviside_sync_phase);
	fftw_free(heaviside_ordered_sync_phase);
	fftw_free(tmp_ordered_time_order_k);
	fftw_free(tmp_heavi_time_order_k);
	fftw_free(tmp_heavi_order_time_order_k);
	#endif
	
	free(amp_norm);
	free(F_k_avg);
	free(Phi_k_dot_avg);
	free(Phi_k_last);
	free(sin_theta_k_avg);
	free(sin_theta_k);
	free(P_k);
	free(R_k);
	#endif
	#ifdef __REALSPACE_STATS
	free(str_func);
	#endif
	#ifdef __TIME
	free(time_array);
	#endif
	fftw_free(u_z);
	fftw_free(RK1);
	fftw_free(RK2);
	fftw_free(RK3);
	fftw_free(RK4);
	fftw_free(u_z_tmp);
	fftw_free(u_z_pad);

	// destroy the complex comound datatype
	#if defined(__MODES) || defined(__TRIAD_STATS) || defined(__TRIAD_ORDER)
	status = H5Tclose(COMPLEX_DATATYPE);
	#endif

	// Close HDF5 handles
	#ifdef __PHASES
	status = H5Sclose( HDF_mem_space[0] );
	status = H5Dclose( HDF_data_set[0] );
	status = H5Sclose( HDF_file_space[0] );
	#endif
	#ifdef __TRIADS
	status = H5Sclose( HDF_mem_space[1] );
	status = H5Dclose( HDF_data_set[1] );
	status = H5Sclose( HDF_file_space[1] );
	#endif
	#ifdef __MODES
	status = H5Sclose( HDF_mem_space[2] );
	status = H5Dclose( HDF_data_set[2] );
	status = H5Sclose( HDF_file_space[2] );
	#endif 
	#ifdef __REALSPACE
	status = H5Sclose( HDF_mem_space[3] );
	status = H5Dclose( HDF_data_set[3] );
	status = H5Sclose( HDF_file_space[3] );
	#endif
	#ifdef __RHS
	status = H5Sclose( HDF_mem_space[4] );
	status = H5Dclose( HDF_data_set[4] );
	status = H5Sclose( HDF_file_space[4] );
	#endif
	#ifdef __GRAD
	status = H5Sclose( HDF_mem_space[5] );
	status = H5Dclose( HDF_data_set[5] );
	status = H5Sclose( HDF_file_space[5] );
	#endif
	#ifdef __TRIAD_ORDER
	#ifdef __PHI_K_DOT
	status = H5Sclose( HDF_mem_space[6] );
	status = H5Dclose( HDF_data_set[6] );
	status = H5Sclose( HDF_file_space[6] );
	#endif
	// status = H5Sclose( HDF_mem_space[7] );
	// status = H5Dclose( HDF_data_set[7] );
	// status = H5Sclose( HDF_file_space[7] );
	#ifdef __ADLER_PHASE_ORDER
	status = H5Sclose( HDF_mem_space[8] );
	status = H5Dclose( HDF_data_set[8] );
	status = H5Sclose( HDF_file_space[8] );
	#endif
	#ifdef __SCALE_PHASE_ORDER
	status = H5Sclose( HDF_mem_space[9] );
	status = H5Dclose( HDF_data_set[9] );
	status = H5Sclose( HDF_file_space[9] );
	#endif
	#ifdef __THETA_TIME_PHASE_ORDER
	status = H5Sclose( HDF_mem_space[10] );
	status = H5Dclose( HDF_data_set[10] );
	status = H5Sclose( HDF_file_space[10] );
	#endif
	#ifdef __OMEGA_K
	status = H5Sclose( HDF_mem_space[11] );
	status = H5Dclose( HDF_data_set[11] );
	status = H5Sclose( HDF_file_space[11] );
	#endif
	// status = H5Sclose( HDF_mem_space[12] );
	// status = H5Dclose( HDF_data_set[12] );
	// status = H5Sclose( HDF_file_space[12] );
	#endif

	// Close pipeline to output file
	status = H5Fclose(HDF_Outputfile_handle);

	// Check if fixed point found 
	#ifdef __FXD_PT_SEARCH__
	if (iters_at_fxd_pt <= 5) EXIT_FLAG = 0;
	#endif

	return EXIT_FLAG;
}

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
	#endif

	#ifdef __RHS
	//-----------------------------//
	//----------- RHS -------------//
	//-----------------------------//
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;             // number of timesteps
	dims[1]      = num_osc;                 // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = num_osc;                 // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = num_osc;                 // 1D chunk of size number of modes

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[4] = H5Screate_simple(dimensions, dims, maxdims);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist5;
	plist5 = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist5, dimensions, chunkdims);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[4] = H5Dcreate(*file_handle, "RHS", H5T_NATIVE_DOUBLE, file_space[4], H5P_DEFAULT, plist5, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims[0] = 1;
	dims[1] = num_osc;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[4] = H5Screate_simple(dimensions, dims, NULL);

	H5Pclose(plist5);
	#endif
	
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

	triads_attr_space = H5Screate_simple(2, adims, NULL);

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
	int NN = 2 * (num_osc - 1);
	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;             // number of timesteps
	dims[1]      = NN;                       // number of collocation points
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = NN;                       // number of collocation points
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = NN;                       // number of collocation points

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
	dims[1] = NN;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[3] = H5Screate_simple(dimensions, dims, NULL);

	H5Pclose(plist4);
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

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[5] = H5Screate_simple(dimensions, dims, maxdims);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist6;
	plist6 = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist6, dimensions, chunkdims);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[5] = H5Dcreate(*file_handle, "RealSpaceGrad", H5T_NATIVE_DOUBLE, file_space[5], H5P_DEFAULT, plist6, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims[0] = 1;
	dims[1] = N;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[5] = H5Screate_simple(dimensions, dims, NULL);

	H5Pclose(plist6);
	#endif
}

hid_t create_complex_datatype() {

	// Declare HDF5 datatype variable
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
	dtype = H5Tcreate(H5T_COMPOUND, sizeof(cmplex));
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

void set_vel_inc_hist_bin_ranges(gsl_histogram** hist_incr, double* u, double* u_grad, int num_osc) {

	// Initialize variables
	int r;
	int N_osc     = num_osc - 1;
	int num_r_inc = 2;
	int r_inc[num_r_inc];
	r_inc[0] = 1;
	r_inc[1] = N_osc;
	double vel_inc;
	double dx = 0.5 / (double)N_osc;
	double varaince;

	// Initialize running stats workspace - used to find min & max bin edges
	gsl_rstat_workspace* vel_inc_stats[num_r_inc + 1];

	
	////////////////////////
	// Compute Increments
	////////////////////////
	// Compute velocity increments
	for (int r_indx = 0; r_indx < num_r_inc; ++r_indx) {
		// Get current incr
		r = r_inc[r_indx]; 
	
		// Initialize stats accumulator
		vel_inc_stats[r_indx] = gsl_rstat_alloc();
		for (int i = 0; i < 2 * N_osc; ++i) {
			// Get current increment
			vel_inc = (u[(i + r) % 2 * N_osc] - u[i]);

			// Add incr to accumulator
			gsl_rstat_add(vel_inc, vel_inc_stats[r_indx]);
		}
	}

	// Get accumulator for the gradient
	vel_inc_stats[num_r_inc] = gsl_rstat_alloc();
	for (int i = 0; i < 2 * N_osc; ++i)
	{
		// Add next gradient value to accum
		gsl_rstat_add(u_grad[i], vel_inc_stats[num_r_inc]);
	}

	//////////////////////
	// Set Bin Ranges
	//////////////////////
	for (int i = 0; i < num_r_inc; ++i) {
		// Get the variance of the smallest incr
		varaince = gsl_rstat_variance(vel_inc_stats[i]);
		if ( (gsl_histogram_set_ranges_uniform(hist_incr[i], -BIN_LIM * sqrt(varaince), BIN_LIM * sqrt(varaince))) != 0 ) {
			fprintf(stderr, "ERROR: unable to set ranges for the GSL histogram: Hist_Incrment[%d]\n", i);
			exit(1);						
		}
	}

	// Get the variance of the gradient
	varaince = gsl_rstat_variance(vel_inc_stats[num_r_inc]);
	if ( (gsl_histogram_set_ranges_uniform(hist_incr[num_r_inc], -BIN_LIM * sqrt(varaince), BIN_LIM * sqrt(varaince))) != 0 ) {
		fprintf(stderr, "ERROR: unable to set ranges for the GSL histogram: %s\n", "VelocityGradient");
		exit(1);						
	}	

	// Free memory
	for (int i = 0; i < num_r_inc + 1; ++i) {
		gsl_rstat_free(vel_inc_stats[i]);
	}
	
}


void compute_real_space_stats(gsl_histogram** hist_incr, gsl_rstat_workspace** incr_stat, double* str_func, double* u, double* u_grad, int num_osc, int max_p) {

	// Initialize variables
	int r;
	int N_osc     = num_osc - 1;
	int num_r_inc = 2;
	int r_inc[num_r_inc];
	r_inc[0] = 1;
	r_inc[1] = N_osc;
	double vel_inc;
	double vel_inc_abs;
	double dx = 0.5 / (double)N_osc;


	////////////////////////
	// Compute Increments
	////////////////////////
	// Compute Velocity Incrments
	for (int r_indx = 0; r_indx < num_r_inc; ++r_indx) {
		// Get current incr
		r = r_inc[r_indx]; 
		
		for (int i = 0; i < 2 * N_osc; ++i) {
			// Get current increment
			vel_inc = u[(i + r) % (2* N_osc)] - u[i];

			// Add current vel inc to appropriate bin
			gsl_histogram_increment(hist_incr[r_indx], vel_inc);

			// Add current vel inc to accumulator
			gsl_rstat_add(vel_inc, incr_stat[r_indx]);
		}
	}
	
	// Compute Gradient Histogram
	for (int i = 0; i < 2 * N_osc; ++i)	{
		// Add current gradient to appropriate bin
		gsl_histogram_increment(hist_incr[num_r_inc], u_grad[i]);
	}

	///////////////////////////////
	// Compute Structure Functions
	///////////////////////////////
	for (int p = 2; p <= max_p; ++p) {
		for (int r = 1; r <= N_osc; ++r) {		
			vel_inc = 0.0;
 			for (int i = 0; i < 2 * N_osc; ++i) {
				// Get current increment
				vel_inc     += pow(u[(i + r) % (2 * N_osc)] - u[i], p);
			}
			// Update structure func
			str_func[(p - 2) * N_osc + (r - 1)]+= vel_inc * dx;
		}
	}

}


int solver(int N, int k0, double a, double b, int iters, int save_step, char* u0) {

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

	// stucture function array
	int max_p     = 6;
	int num_stats = 0;
	double *str_func     = (double *)malloc(sizeof(double) * (max_p - 2 + 1) * (num_osc - 1));
	mem_chk(str_func, "str_func");
	for (int i = 0; i < (max_p - 2 + 1); ++i) {
		for (int j = 0; j < (num_osc - 1); ++j) {
			str_func[i * (num_osc - 1) + j] = 0.0;
		}
	}
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
	// create fftw3 plans objects
	fftw_plan fftw_plan_r2c_pad, fftw_plan_c2r_pad;
	
	// create plans - ensure no overwriting - fill arrays after
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
	hid_t HDF_file_space[6];
	hid_t HDF_data_set[6];
	hid_t HDF_mem_space[6];

	// get output file name
	char output_file_name[512];
	get_output_file_name(output_file_name, N, k0, a, b, u0, ntsteps, trans_iters);

	// Create complex datatype for hdf5 file if modes are being recorded
	#if defined(__MODES) || defined(__TRIAD_STATS)
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
	mem_chk(time_array, "time_array");
	#ifdef __TRIADS
	double* phase_order_R   = (double* )malloc(sizeof(double) * (num_save_steps));
	double* phase_order_Phi = (double* )malloc(sizeof(double) * (num_save_steps));
	mem_chk(phase_order_R, "phase_order_R");
	mem_chk(phase_order_Phi, "phase_order_Phi");	
	#endif

	// Write initial condition if transient iterations are not being performed
	#ifndef __TRANSIENTS
	// Write Initial condition for phases
	write_hyperslab_data(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], H5T_NATIVE_DOUBLE, phi, "phi", num_osc, 0);

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

	

		////////////
		// Print to file
		////////////
		if ((iter > trans_iters) && (iter % save_step == 0)) {
			if (save_data_indx == 0) {
				begin = clock();
			}
			
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

			#if defined(__MODES) || defined(__GRAD) || defined(__REALSPACE) || defined(__REALSPACE_STATS)
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
			// fftw_execute_dft_c2r((*plan_c2r_pad), u_z_tmp, u_tmp);
			for (int i = 0; i < N; ++i)	{
				u[i] /= (double) N;  // Normalize inverse transfom
			}
			#endif

			#if defined(__GRAD) || defined(__REALSPACE_STATS) 
			// transform back to Real Space
			fftw_execute_dft_c2r(fftw_plan_c2r, u_z_grad, u_grad);
			for (int i = 0; i < N; ++i)	{
				u_grad[i] /= (double) N;  // Normalize inverse transfom
			}
			#endif

			#ifdef __REALSPACE_STATS			
			// If first non-transient iteration - set bin edges
			if ((trans_iters > 0) && (save_data_indx == 0)) {
				set_vel_inc_hist_bin_ranges(vel_inc_hist, u, u_grad, num_osc);
			}

			compute_real_space_stats(vel_inc_hist, vel_inc_stats, str_func, u, u_grad, num_osc, max_p);
			num_stats++;
			#endif

			#ifdef __GRAD
			// Write real space
			write_hyperslab_data(HDF_file_space[5], HDF_data_set[5], HDF_mem_space[5], H5T_NATIVE_DOUBLE, u_grad, "u_grad", N, save_data_indx);
			#endif
			#ifdef __REALSPACE
			// Write real space
			write_hyperslab_data(HDF_file_space[3], HDF_data_set[3], HDF_mem_space[3], H5T_NATIVE_DOUBLE, u, "u", N, save_data_indx);
			#endif

			// save time and phase order parameter
			time_array[save_data_indx]  = iter * dt;

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

	// Finish timing
	clock_t end = clock();

	// calculate execution time
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("\n\n\tTotal Execution Time: %20.16lf\n\n", time_spent);



	// ------------------------------
	//  Write 1D Arrays Using HDF5Lite
	// ------------------------------
	const hid_t D1 = 1;
	hid_t D1dims[D1];

	// Write amplitudes
	D1dims[0] = num_osc;
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "Amps", D1, D1dims, H5T_NATIVE_DOUBLE, amp)) < 0) {
		printf("\n\n!!Failed to make - Amps - Dataset!!\n\n");
	}
	
	// Wtie time
	D1dims[0] = num_save_steps;
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "Time", D1, D1dims, H5T_NATIVE_DOUBLE, time_array)) < 0) {
		printf("\n\n!!Failed to make - Time - Dataset!!\n\n");
	}
	
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
	const hid_t D2 = 2;
	hid_t D2dims[D2];
	D2dims[0] = num_r_inc + 1;
	D2dims[1] = 4;
	if ( (H5LTmake_dataset(HDF_Outputfile_handle, "VelIncStats", D2, D2dims, H5T_NATIVE_DOUBLE, vel_inc_stats_data)) < 0) {
		printf("\n\n!!Failed to make - %s - Dataset!!\n\n", "VelIncStats");
	}	

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
	H5Dwrite(triad_cent_dset, COMPLEX_DATATYPE, H5S_ALL, H5S_ALL, H5P_DEFAULT, triad_centroid);


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

    herr_t status = H5Awrite(triad_cent, H5T_NATIVE_INT, triad_cent_dims);

	// close the created property list
	status = H5Aclose(triad_cent);
    status = H5Sclose(triad_cent_space);
    // Close datasets and spaces
	H5Dclose(triad_cent_dset);
	H5Sclose(triad_cent_dspace);
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
	#ifdef __REALSPACE_STATS
	free(str_func);
	#endif
	free(time_array);
	fftw_free(u_z);
	fftw_free(RK1);
	fftw_free(RK2);
	fftw_free(RK3);
	fftw_free(RK4);
	fftw_free(u_z_tmp);
	fftw_free(u_z_pad);


	// Close HDF5 handles
	#ifdef __PHASES
	H5Sclose( HDF_mem_space[0] );
	H5Dclose( HDF_data_set[0] );
	H5Sclose( HDF_file_space[0] );
	#endif
	#ifdef __TRIADS
	H5Sclose( HDF_mem_space[1] );
	H5Dclose( HDF_data_set[1] );
	H5Sclose( HDF_file_space[1] );
	#endif
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
	#ifdef __RHS
	H5Sclose( HDF_mem_space[4] );
	H5Dclose( HDF_data_set[4] );
	H5Sclose( HDF_file_space[4] );
	#endif
	#ifdef __GRAD
	H5Sclose( HDF_mem_space[5] );
	H5Dclose( HDF_data_set[5] );
	H5Sclose( HDF_file_space[5] );
	#endif

	// Close pipeline to output file
	H5Fclose(HDF_Outputfile_handle);

	// Check if fixed point found 
	#ifdef __FXD_PT_SEARCH__
	if (iters_at_fxd_pt <= 5) EXIT_FLAG = 0;
	#endif

	return EXIT_FLAG;
}

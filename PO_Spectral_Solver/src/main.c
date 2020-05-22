 // Enda Carroll
// May 2020
// Main file for calling the pseudospectral solver for the 1D Burgers equation

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
#include "solver.h"





// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	// ------------------------------
	//  Variable Definitions
	// ------------------------------
	// Collocation points
	int N = atoi(argv[1]);

	// Number of modes
	int num_osc = (N / 2) + 1; 

	// padded array size
	int M = 2 * N;

	// Forcing wavenumber
	int k0 = 0;
	int kmin = k0 + 1;
	int kmax = num_osc - 1;
	
	// Spectrum vars
	double a = 1.0;
	double b = 0.0;
	double cutoff = ((double) num_osc - 1.0) / 2.0;
	
	// spatial variables
	double leftBoundary  = 0.0;
	double rightBoundary = 2.0*M_PI;
	double dx            = (rightBoundary - leftBoundary) / (double)N;


	// ------------------------------
	//  Allocate memory
	// ------------------------------
	// wavenumbers
	int* kx;
	kx = (int* ) malloc(num_osc * sizeof(int));

	// Oscillator arrays
	double* amp;
	amp = (double* ) malloc(num_osc * sizeof(double));
	double* phi;
	phi = (double* ) malloc(num_osc * sizeof(double));

	// soln arrays
	double* u;
	u = (double* ) malloc(N*sizeof(double));

	fftw_complex* u_z;
	u_z	= (fftw_complex* ) fftw_malloc(num_osc * sizeof(fftw_complex));

	// padded solution arrays
	double* u_pad;
	u_pad = (double* ) malloc(M * sizeof(double));

	fftw_complex* u_z_pad;
	u_z_pad	= (fftw_complex* ) fftw_malloc(2 * num_osc * sizeof(fftw_complex));

	// Triad phases array
	int k_range  = kmax - kmin + 1;
	int k1_range = (int)((kmax - kmin + 1)/ 2.0);

	double* triads;
	triads = (double* )malloc(k_range * k1_range * sizeof(double));
	// initialize triad array to handle empty elements
	for (int i = 0; i < k_range; ++i) {
		int tmp = i * k1_range;
		for (int j = 0; j < k1_range; ++j) {
			int indx = tmp + j;
			triads[indx] = -10.0;
		}
	}	

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

	// temporary memory to store stages
	fftw_complex* u_z_tmp;
	u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));




	// ------------------------------
	//  Create FFTW plans
	// ------------------------------
	// create fftw3 plans objects
	fftw_plan fftw_plan_r2c, fftw_plan_c2r;

	// create plans - ensure no overwriting - fill arrays after
	fftw_plan_r2c = fftw_plan_dft_r2c_1d(M, u_pad, u_z_pad, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(M, u_z_pad, u_pad, FFTW_PRESERVE_INPUT);



	// ------------------------------
	//  Generate initial conditions
	// ------------------------------
	initial_condition(phi, amp, kx, num_osc, k0, cutoff, a, b);

	for (int i = 0; i < num_osc; ++i) {
		printf("k[%d]: %d | a: %5.15lf   p: %5.16lf   | u_z[%d]: %5.16lf  %5.16lfI \n", i, kx[i], amp[i], phi[i], i, creal(amp[i] * cexp(I * phi[i])), cimag(amp[i] * cexp(I * phi[i])));
	}
	printf("\n");



	// ------------------------------
	//  Get Timestep & Time variables
	// ------------------------------
	double dt;
	dt = get_timestep(amp, fftw_plan_c2r, fftw_plan_r2c, kx, N, num_osc, k0);

	// time varibales
	int ntsteps = 1e3; 
	double t0   = 0.0;
	double T    = t0 + ntsteps * dt;



	// // ------------------------------
	// //  HDF5 File Create
	// // ------------------------------
	// // create hdf5 file identifier handle
	// hid_t HDF_file_handle;


	// // define filename - const because it doesnt change
	// const char* output_file_name = "./output/Runtime_Data.h5";

	// // create datafile - H5F_ACC_TRUNC overwrites file if it exists already
	// HDF_file_handle = H5Fcreate(output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	// /* create compound hdf5 datatype for complex modes */
	// // create the C stuct to store the complex datatype
	// typedef struct compound_complex {
	// 	double re;
	// 	double im;
	// } compound_complex;

	// // declare and intialize the new complex datatpye
	// compound_complex complex_data;
	// complex_data.re = 0.0;
	// complex_data.im = 0.0;

	// // create the new HDF5 compound datatpye using the new complex datatype above
	// // use this id to write/read the complex modes to/from file later
	// hid_t comp_id; 
	// comp_id = H5Tcreate(H5T_COMPOUND, sizeof(complex_data));

	// // insert each of the members of the complex datatype struct into the new HDF5 compound datatype
	// // using the offsetof() function to find the offset in bytes of the field of the complex struct type
	// H5Tinsert(comp_id, "r", offsetof(compound_complex, re), H5T_NATIVE_DOUBLE);
	// H5Tinsert(comp_id, "i", offsetof(compound_complex, im), H5T_NATIVE_DOUBLE);
	
	
	// /* use hdf5_hl library to create datasets */
	// // define dimensions and dimension sizes
	// hsize_t HDF_D1ndims = 1;
	// hsize_t D1dims[HDF_D1ndims];
	// hsize_t HDF_D2ndims = 2;
	// hsize_t D2dims[HDF_D2ndims];


	// // ------------------------------
	// //  HDF5 Hyperslab space creation
	// // ------------------------------
	// // create hdf5 handle identifiers for hyperslabing the full evolution data
	// hid_t HDF_file_space, HDF_data_set, HDF_data_set_real, HDF_mem_space;

	// // create hdf5 dimension arrays for creating the hyperslabs
	// static const int dimensions = 2;
	// hsize_t dims[dimensions];      // array to hold dims of full evolution data
	// hsize_t maxdims[dimensions];   // array to hold max dims of full evolution data
	// hsize_t chunkdims[dimensions]; // array to hold dims of the hyperslab chunks

	// // initialize the hyperslab arrays
	// dims[0]      = ceil((T - t0) / dt) + 1; // number of timesteps
	// dims[1]      = num_osc;                 // number of modes
	// maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	// maxdims[1]   = N;                 // same as before = number of modes
	// chunkdims[0] = 1;                       // 1D chunk to be saved 
	// chunkdims[1] = N;                 // 1D chunk of size number of modes


	// // create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	// HDF_file_space = H5Screate_simple(dimensions, dims, maxdims);


	// // must create a propertly list to enable data chunking due to max time dimension being unlimited
	// // create property list 
	// hid_t plist;
	// plist = H5Pcreate(H5P_DATASET_CREATE);

	// // using this property list set the chuncking - stores the chunking info in plist
	// H5Pset_chunk(plist, dimensions, chunkdims);


	// // Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	// HDF_data_set = H5Dcreate(HDF_file_handle, "ComplexModesFull", comp_id, HDF_file_space, H5P_DEFAULT, plist, H5P_DEFAULT);
	// HDF_data_set_real = H5Dcreate(HDF_file_handle, "RealModesFull", H5T_NATIVE_DOUBLE, HDF_file_space, H5P_DEFAULT, plist, H5P_DEFAULT);


	// // create the memory space for the slab
	// dims[0] = 1;
	// dims[1] = N;

	// // setting the max dims to NULL defaults to same size as dims
	// HDF_mem_space = H5Screate_simple(dimensions, dims, NULL);


	// // close the created property list
	// H5Pclose(plist);



	// ------------------------------
	//  Write Initial Conditions to File
	// ------------------------------
	/* write initial condition to hypslab */
	// Create the necessary arrays to insert initial condition into correct spot
	// hsize_t start_index[dimensions]; // stores the index in the hyperslabbed dataset to start writing to
	// hsize_t count[dimensions];       // stores the size of hyperslab to write to the dataset

	// count[0]       = 1;				 // 1D slab so first dim is 1
	// count[1]       = N;		 // 1D slab of size number of modes
	// start_index[0] = 0;			     // set the starting row index to first position in the global dataset
	// start_index[1] = 0;				 // set column index to 0 to start writing from the first column


	// // select appropriate hyperslab 
	// H5Sselect_hyperslab(HDF_file_space, H5S_SELECT_SET, start_index, NULL, count, NULL);
	

	// // then write the current modes to this hyperslab
	// H5Dwrite(HDF_data_set, comp_id, HDF_mem_space, HDF_file_space, H5P_DEFAULT, u_z);
	// H5Dwrite(HDF_data_set_real, H5T_NATIVE_DOUBLE, HDF_mem_space, HDF_file_space, H5P_DEFAULT, u);


	

	


	// ------------------------------
	//  Integration & Looping Params
	// ------------------------------
	int iter = 1;
	double t = 0.0;	


	// ------------------------------
	//  Begin Integration
	// ------------------------------
	while (t < 1*dt) {

		// Construct the modes
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * phi[i]);
		}

		// Print Update - Energy and Enstrophy
		if (iter % (int)(ntsteps * 0.1) == 0) {
			printf("Iter: %d/%d | t = %4.4lf | Energy = %4.8lf,   Enstrophy = %4.8lf\n", iter, ntsteps, t, system_energy(u_z, N), system_enstrophy(u_z, kx, N));
		}
		


		//////////////
		// STAGES
		//////////////
		/*---------- STAGE 1 ----------*/
		// find RHS first and then update stage
		po_rhs(RK1, u_z_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A21 * dt * RK1[i]));
		}

		// for (int i = 0; i < num_osc; ++i) {
		// 	printf("RHS[%d]: %5.16lf \n", i, RK1[i]);
		// }
		// printf("\n\n");

		/*---------- STAGE 2 ----------*/
		// find RHS first and then update stage
		po_rhs(RK2, u_z_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A32 * dt * RK2[i]));
		}

		/*---------- STAGE 3 ----------*/
		// find RHS first and then update stage
		po_rhs(RK3, u_z_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A43 * dt * RK3[i]));
		}

		/*---------- STAGE 4 ----------*/
		// find RHS first and then update 
		po_rhs(RK4, u_z_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);
		

		//////////////
		// Update
		//////////////
		for (int i = 0; i < num_osc; ++i) {
			phi[i] = phi[i] + (dt * B1) * RK1[i] + (dt * B2) * RK2[i] + (dt * B3) * RK3[i] + (dt * B4) * RK4[i];  
		}


		//////////////
		// Print to file
		//////////////
		// if ((stats == 0 && iter % save_step == 0) || (stationary == 1 && iter % save_step == 0)) {
		// 	//update the hyperslab starting index
		// 	start_index[0] = iter;

		// 	// select appropriate hyperslab 
		// 	H5Sselect_hyperslab(HDF_file_space, H5S_SELECT_SET, start_index, NULL, count, NULL);

		// 	// then write the current modes to this hyperslab
		// 	H5Dwrite(HDF_data_set, comp_id, HDF_mem_space, HDF_file_space, H5P_DEFAULT, u_z);

		// 	// transform back to real and write to hyper slab
		// 	fftw_execute(fftw_plan_c2r);
		// 	for (int i = 0; i < N; ++i)	{
		// 		u[i] /= N;
		// 	}
		// 	H5Dwrite(HDF_data_set_real, H5T_NATIVE_DOUBLE, HDF_mem_space, HDF_file_space, H5P_DEFAULT, u);
						
		// 	// Compute the triads
		// 	triad_phases(triads, &triad_phase_order, phi, kmin, kmax);

		// 	save_data_indx++;
		// }
				


		// increment
		t   = iter*dt;
		iter++;
	}


	// ------------------------------
	//  Clean Up
	// ------------------------------
	// destroy fftw plans
	fftw_destroy_plan(fftw_plan_r2c);
	fftw_destroy_plan(fftw_plan_c2r);


	// free memory
	free(kx);
	free(u);
	free(u_pad);
	free(triads);
	fftw_free(u_z);
	fftw_free(RK1);
	fftw_free(RK2);
	fftw_free(RK3);
	fftw_free(RK4);
	fftw_free(u_z_tmp);
	fftw_free(u_z_pad);


	// // close hdf5 datafile
	// H5Fclose(HDF_file_handle);
	

	return 0;
}
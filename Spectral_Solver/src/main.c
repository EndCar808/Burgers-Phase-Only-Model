// Enda Carroll
// Sept 2019
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


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "utils.h"
// #include "forcing.h"





// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	// ------------------------------
	//  Variable Definitions
	// ------------------------------
	// Domain variables
	int N = atoi(argv[1]);
	
	// spatial variables
	double leftBoundary  = 0.0;
	double rightBoundary = 2.0*M_PI;
	double dx            = (rightBoundary - leftBoundary) / (double)N;

	
	// time varibales
	double t0 = 0.0;
	double T  = 1.0;
	double dt = 1e-3;


	// wavenumbers
	int* kx;
	kx = (int* )malloc(N*sizeof(int));


	// soln vectors
	double* u;
	u = (double* )malloc(N*sizeof(double));

	double complex* u_z;
	u_z	= (double complex* )fftw_malloc((N/2 + 1)*sizeof(double complex));



	// ------------------------------
	//  HDF5 File Create
	// ------------------------------
	// create hdf5 file identifier handle
	hid_t HDF_file_handle;


	// define filename - const because it doesnt change
	const char* output_file_name = "./output/Runtime_Data.h5";

	// create datafile - H5F_ACC_TRUNC overwrites file if it exists already
	HDF_file_handle = H5Fcreate(output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	/* create compound hdf5 datatype for complex modes */
	// create the C stuct to store the complex datatype
	typedef struct compound_complex {
		double re;
		double im;
	} compound_complex;

	// declare and intialize the new complex datatpye
	compound_complex complex_data;
	complex_data.re = 0.0;
	complex_data.im = 0.0;

	// create the new HDF5 compound datatpye using the new complex datatype above
	// use this id to write/read the complex modes to/from file later
	hid_t comp_id; 
	comp_id = H5Tcreate(H5T_COMPOUND, sizeof(complex_data));

	// insert each of the members of the complex datatype struct into the new HDF5 compound datatype
	// using the offsetof() function to find the offset in bytes of the field of the complex struct type
	H5Tinsert(comp_id, "r", offsetof(compound_complex, re), H5T_NATIVE_DOUBLE);
	H5Tinsert(comp_id, "i", offsetof(compound_complex, im), H5T_NATIVE_DOUBLE);
	
	
	/* use hdf5_hl library to create datasets */
	// define dimensions and dimension sizes
	hsize_t HDF_D1ndims = 1;
	hsize_t D1dims[HDF_D1ndims];
	hsize_t HDF_D2ndims = 2;
	hsize_t D2dims[HDF_D2ndims];



	
	// ------------------------------
	//  FFTW plans
	// ------------------------------
	// create fftw3 plans objects
	fftw_plan fftw_plan_r2c, fftw_plan_c2r;

	// create plans - ensure no overwriting - fill arrays after
	fftw_plan_r2c = fftw_plan_dft_r2c_1d(N, u, u_z, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(N, u_z, u, FFTW_PRESERVE_INPUT);




	// ------------------------------
	//  Generate initial conditions
	// ------------------------------
	initial_condition(u, u_z, fftw_plan_r2c, dx, N);


	// for (int i = 0; i < N/2 + 1; ++i)	{
	// 	printf("u[%d] = %5.13lf, u_z[%d] = %5.13lf + %5.13lfI\n", i, u[i], i, creal(u_z[i]), cimag(u_z[i]));
	// }
	

	// ------------------------------
	//  CFL condition
	// ------------------------------
	// find max u
	int max = max_indx_d(u, N);

	// printf("\nindex = %d, max = %10.16lf\n", max, u[max]);

	// set dt using CFL
	double u_max = fabs(u[max]);
	dt           = (1.0*dx) / u_max; 



	// printf("\nu_max = %10.16lf\n", u_max);
	// printf("\nCFL  dt = %10.16lf\n", dt);
	


	// ------------------------------
	//  HDF5 Hyperslab space creation
	// ------------------------------
	// create hdf5 handle identifiers for hyperslabing the full evolution data
	hid_t HDF_file_space, HDF_data_set, HDF_mem_space;

	// create hdf5 dimension arrays for creating the hyperslabs
	static const int dimensions = 2;
	hsize_t dims[dimensions];      // array to hold dims of full evolution data
	hsize_t maxdims[dimensions];   // array to hold max dims of full evolution data
	hsize_t chunkdims[dimensions]; // array to hold dims of the hyperslab chunks

	// initialize the hyperslab arrays
	dims[0]      = ceil((T - t0) / dt) + 1; // number of timesteps
	dims[1]      = N/2 + 1;                 // number of modes
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = N/2 + 1;                 // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = N/2 + 1;                 // 1D chunk of size number of modes


	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	HDF_file_space = H5Screate_simple(dimensions, dims, maxdims);


	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist;
	plist = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist, dimensions, chunkdims);


	// Create the dataset in the previouosly create datafile - using the chunk enabled property list and new compound datatype
	HDF_data_set = H5Dcreate(HDF_file_handle, "ComplexModesFull", comp_id, HDF_file_space, H5P_DEFAULT, plist, H5P_DEFAULT);


	// create the memory space for the slab
	dims[0] = 1;
	dims[1] = N/2 + 1;

	// setting the max dims to NULL defaults to same size as dims
	HDF_mem_space = H5Screate_simple(dimensions, dims, NULL);


	// close the created property list
	H5Pclose(plist);



	// ------------------------------
	// Viscostiy
	// ------------------------------
	double nu = 0.001;




	// ------------------------------
	//  Runge-Kutta  Variables
	// ------------------------------
	// Define RK4 variables
	static double C2 = 0.5, A21 = 0.5, \
				  C3 = 0.5,           A32 = 0.5, \
				  C4 = 1.0,                      A43 = 1.0, \
				            B1 = 1.0/6.0, B2 = 1.0/3.0, B3 = 1.0/3.0, B4 = 1.0/6.0; 

	// Memory fot the four RHS evaluations in the stages 
	double complex* RK1, *RK2, *RK3, *RK4;
	RK1 = (double complex* )fftw_malloc((N/2 + 1)*sizeof(double complex));
	RK2 = (double complex* )fftw_malloc((N/2 + 1)*sizeof(double complex));
	RK3 = (double complex* )fftw_malloc((N/2 + 1)*sizeof(double complex));
	RK4 = (double complex* )fftw_malloc((N/2 + 1)*sizeof(double complex));

	// temporary memory to store stages
	double complex* stage_tmp;
	stage_tmp = (double complex* )fftw_malloc((N/2 + 1)*sizeof(double complex));



	// ------------------------------
	//  Integration & Looping Params
	// ------------------------------
	int iter = 1;
	double t = 0.0;
	hsize_t start_index[dimensions]; // stores the index in the hyperslabbed dataset to start writing to
	hsize_t count[dimensions];       // stores the size of hyperslab to write to the dataset

	count[0]       = 1;				 // 1D slab so first dim is 1
	count[1]       = N/2 + 1;		 // 1D slab of size number of modes
	start_index[0] = iter;			 // set to iteration counter for now - i.e. index is current iteration
	start_index[1] = 0;				 // set column index to 0 to start writing from the first column



	// ------------------------------
	//  Begin Integration
	// ------------------------------
	while (t < T) {

		// Print Energy and Enstrophy
		printf("Iter: %d | t = %4.4lf | Energy = %4.8lf,   Enstrophy = %4.8lf\n", iter, t, system_energy(u_z, N), system_enstrophy(u_z, kx, N));

		//////////////
		// STAGES
		//////////////
		/*---------- STAGE 2 ----------*/
		// find RHS first and then update stage
		deriv(u_z, RK1, nu, kx, &fftw_plan_r2c, &fftw_plan_c2r, N);
		for (int i = 0; i < N/2 + 1; ++i) {
			stage_tmp[i] = u_z[i] + (dt*A21)*RK1[i];
		}
		/*---------- STAGE 3 ----------*/
		// find RHS first and then update stage
		deriv(stage_tmp, RK2, nu, kx, &fftw_plan_r2c, &fftw_plan_c2r, N);
		for (int i = 0; i < N/2 + 1; ++i) {
			stage_tmp[i] = u_z[i] + (dt*A32)*RK2[i];
		}
		/*---------- STAGE 4 ----------*/
		// find RHS first and then update stage
		deriv(stage_tmp, RK3, nu, kx, &fftw_plan_r2c, &fftw_plan_c2r, N);
		for (int i = 0; i < N/2 + 1; ++i) {
			stage_tmp[i] = u_z[i] + (dt*A43)*RK3[i];
		}
		deriv(stage_tmp, RK4, nu, kx, &fftw_plan_r2c, &fftw_plan_c2r, N);
		

		//////////////
		// Update
		//////////////
		for (int i = 0; i < N/2 + 1; ++i) {
			u_z[i] = u_z[i] + (dt*B1)*RK1[i] + (dt*B2)*RK2[i] + (dt*B3)*RK3[i] + (dt*B4)*RK4[i];  
		}



		//////////////
		// Print to file
		//////////////
		//update the hyperslab starting index
		start_index[0] = iter;

		// select appropriate hyperslab 
		H5Sselect_hyperslab(HDF_file_space, H5S_SELECT_SET, start_index, NULL, count, NULL);

		// then write the current modes to this hyperslab
		H5Dwrite(HDF_data_set, comp_id, HDF_mem_space, HDF_file_space, H5P_DEFAULT, u_z);
		


		// increment
		t    += dt;
		iter += 1;
	}


	// ------------------------------
	//  Write to file
	// ------------------------------
	// use hdf5_hl functionality to make the modes dataset using the compound datatype
	D1dims[0] = N/2 + 1;
	H5LTmake_dataset (HDF_file_handle, "ComplexModesSnapshot", HDF_D1ndims, D1dims, comp_id, u_z);





	// ------------------------------
	//  Clean Up
	// ------------------------------
	// destroy fftw plans
	fftw_destroy_plan(fftw_plan_r2c);
	fftw_destroy_plan(fftw_plan_c2r);


	// free memory
	free(kx);
	free(u);
	fftw_free(u_z);
	fftw_free(RK1);
	fftw_free(RK2);
	fftw_free(RK3);
	fftw_free(RK4);
	fftw_free(stage_tmp);


	// close hdf5 datafile
	H5Fclose(HDF_file_handle);
	

	return 0;
}
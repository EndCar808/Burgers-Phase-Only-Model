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
	// using the offsetof() function to find the offset in bytes of the field of the complex struct
	H5Tinsert(comp_id, "r", offsetof(complex_data, re), H5T_NATIVE_DOUBLE);
	H5Tinsert(comp_id, "i", offsetof(complex_data, im), H5T_NATIVE_DOUBLE);
	
	
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
	//  Begin Integration
	// ------------------------------
	int iter = 1;
	double t = 0.0;
	while (t < dt*1.0) {

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


		// increment
		t    += dt;
		iter += 1;
	}


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